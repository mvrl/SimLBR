import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision, BinaryF1Score

from .backbones import VisionBackbone


class SimLBR(LightningModule):
    def __init__(
        self,
        backbone: str = "dinov3",
        lr: float = 1e-4,
        wt_decay: float = 1e-2,
        activation: str = "relu",
        hidden_layers: int = 1,
        dropout: float = 0.3,
        lbr: bool = False,
        lbrdist=(0.5, 0.8),
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lbr = lbr
        self.lbrdist = self._validate_dist(lbrdist)
        self.backbone = VisionBackbone(model_name=backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.input_dims = self.backbone.feat_dim
        self.classifier = self._build_classifier(
            input_dims=self.input_dims,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.val_ap = BinaryAveragePrecision()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()
        self.test_ap = BinaryAveragePrecision()

        print(f"LBR enabled: {self.lbr}")
        if self.lbr:
            print(f"LBR alpha distribution: U{self.lbrdist}")

    @staticmethod
    def _validate_dist(dist) -> tuple[float, float]:
        if not isinstance(dist, (tuple, list)) or len(dist) != 2:
            raise ValueError("lbrdist must be a list/tuple of length 2.")
        low, high = float(dist[0]), float(dist[1])
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError("lbrdist must satisfy 0 <= low <= high <= 1.")
        return low, high

    @staticmethod
    def _activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation function: {name}")

    def _build_classifier(
        self,
        input_dims: int,
        hidden_layers: int,
        activation: str,
        dropout: float,
    ) -> nn.Module:
        layers = []
        current_dims = input_dims
        for _ in range(hidden_layers):
            next_dims = max(current_dims // 2, 1)
            layers.append(nn.Linear(current_dims, next_dims))
            layers.append(self._activation(activation))
            layers.append(nn.Dropout(dropout))
            current_dims = next_dims
            if current_dims == 1:
                break
        layers.append(nn.Linear(current_dims, 1))
        return nn.Sequential(*layers)

    def _sample_alpha(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        low, high = self.lbrdist
        return torch.empty(size, 1, device=device, dtype=dtype).uniform_(low, high)

    def extract_cls(self, x):
        with torch.no_grad():
            return self.backbone(x)["cls_token"]

    def apply_lbr(
        self,
        anchor_cls_tokens: torch.Tensor,
        pair_cls_tokens: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if not self.lbr:
            return anchor_cls_tokens

        fake_mask = labels == 1
        if not fake_mask.any():
            return anchor_cls_tokens

        mixed_cls_tokens = anchor_cls_tokens.clone()
        alpha = self._sample_alpha(
            int(fake_mask.sum().item()),
            anchor_cls_tokens.device,
            anchor_cls_tokens.dtype,
        )
        fake_pair_cls_tokens = (
            pair_cls_tokens[fake_mask]
            if pair_cls_tokens.size(0) == labels.size(0)
            else pair_cls_tokens
        )
        mixed_cls_tokens[fake_mask] = (
            alpha * fake_pair_cls_tokens
            + (1.0 - alpha) * anchor_cls_tokens[fake_mask]
        )
        return mixed_cls_tokens

    def forward(self, x):
        cls_tokens = self.extract_cls(x)
        return self.classifier(cls_tokens)

    def shared_step(self, batch, apply_lbr: bool):
        anchor = batch["anchor"]
        labels = batch["label"].float()

        cls_tokens = self.extract_cls(anchor)
        fake_mask = labels == 1
        if apply_lbr and self.lbr and fake_mask.any():
            pair_cls_tokens = self.extract_cls(batch["pair"][fake_mask])
            cls_tokens = self.apply_lbr(cls_tokens, pair_cls_tokens, labels)

        logits = self.classifier(cls_tokens).squeeze(1)
        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, apply_lbr=True)
        self.train_acc.update(preds, labels.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, apply_lbr=False)
        self.val_acc.update(preds, labels.int())
        self.val_f1.update(preds, labels.int())
        self.val_ap.update(preds, labels.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)
        self.log("val_ap", self.val_ap, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        self.preds = []
        self.labels = []

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.shared_step(batch, apply_lbr=False)
        self.test_acc.update(preds, labels.int())
        self.test_f1.update(preds, labels.int())
        self.test_ap.update(preds, labels.int())
        self.log("predict_loss", loss, prog_bar=True)
        self.log("predict_acc", self.test_acc, prog_bar=True)
        self.log("predict_f1", self.test_f1, prog_bar=True)
        self.log("predict_ap", self.test_ap, prog_bar=True)
        self.preds.append(preds)
        self.labels.append(labels)
        return preds

    def on_train_epoch_end(self):
        self.train_acc.reset()
        self.log("Epoch", self.current_epoch)

    def on_validation_epoch_end(self):
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_ap.reset()

    def on_test_epoch_end(self):
        print(f"Test Accuracy: {self.test_acc.compute()}")
        print(f"Test F1 Score: {self.test_f1.compute()}")
        print(f"Test Average Precision: {self.test_ap.compute()}")

    def on_test_end(self):
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_ap.reset()
        self.all_preds = torch.cat(self.preds)
        self.all_labels = torch.cat(self.labels)
        prediction_bin = (self.all_preds > 0.5).int()
        self.correct = (prediction_bin == self.all_labels.int()).int()

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.wt_decay,
        )
        return {"optimizer": optimizer}
