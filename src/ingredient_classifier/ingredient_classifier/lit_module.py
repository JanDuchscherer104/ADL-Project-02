from enum import Enum, auto
from typing import Any, Dict, Tuple, Type, Union

import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision.models as models
from pydantic import Field
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR

from litutils import BaseConfig

from .alexnet import AlexNetParams


class OptimizerConfig(BaseConfig):
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    lr_max: float = 0.01
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 1e4
    pct_start: float = 0.3
    anneal_strategy: str = "cos"


class ImgClassifierParams(BaseConfig["LitImageClassifierModule"]):
    target: Type["LitImageClassifierModule"] = Field(
        default_factory=lambda: LitImageClassifierModule
    )

    num_classes: int = 46
    model: "ModelType" = Field(default_factory=lambda: ModelType.ALEXNET)
    batch_size: int = 32
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    train_head_only: bool = False


# Define the model type enumeration
class ModelType(Enum):
    ALEXNET = auto()
    RESNET50 = auto()
    VISION_TRANSFORMER = auto()

    @classmethod
    def setup_target(cls, params: "ImgClassifierParams") -> nn.Module:
        match params.model:
            case cls.ALEXNET:
                return AlexNetParams(num_classes=params.num_classes).setup_target()
            case cls.RESNET50:
                # Instantiate ResNet50 with ImageNet weights (V1: acc@1 76.13, V2: acc@1 80.86)
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                if params.train_head_only:
                    for param in model.parameters():
                        param.requires_grad = False

                # self.fc = nn.Linear(512 * block.expansion, num_classes)
                model.fc = nn.Linear(model.fc.in_features, params.num_classes)
                return model
            case cls.VISION_TRANSFORMER:
                # Instantiate Vision Transformer with ImageNet weights
                model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
                # if representation_size is None:
                #     heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
                # ...
                # self.heads = nn.Sequential(heads_layers)
                # Use representation_size = None, as it is the default value
                if params.train_head_only:
                    for param in model.parameters():
                        param.requires_grad = False
                model.heads = nn.Linear(model.hidden_dim, params.num_classes)
                return model
            case _:
                raise NotImplementedError(f"Model type {params.model} not implemented")


# Define the classifier model
class LitImageClassifierModule(pl.LightningModule):
    params: ImgClassifierParams

    def __init__(self, params: ImgClassifierParams):
        super().__init__()
        self.params = params
        self.model = ModelType.setup_target(self.params)
        self.loss_fn = nn.CrossEntropyLoss()

        # Define metrics using torchmetrics
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.params.num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.params.num_classes
        )
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            num_classes=self.params.num_classes,
            task="multiclass",
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log training metrics
        acc = self.train_accuracy(y_hat, y)  # applies argmax internally
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log("train_accuracy", acc, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Log validation metrics
        acc = self.val_accuracy(y_hat, y)
        self.log(
            "val_loss", loss, prog_bar=True, on_epoch=True, on_step=False, logger=True
        )
        self.log(
            "val_accuracy",
            acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        # Update confusion matrix
        # self.confusion_matrix.update(y_hat, y)

    def on_validation_epoch_end(self):
        # Get confusion matrix
        # cm = self.confusion_matrix.compute().cpu().numpy()

        # Create figure
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
        # plt.xlabel("Predicted")
        # plt.ylabel("True")
        # plt.title("Confusion Matrix")

        # # Log to wandb if using WandbLogger
        # if isinstance(self.logger, WandbLogger):
        #     self.logger.experiment.log(
        #         {"confusion_matrix": wandb.Image(plt), "global_step": self.global_step}
        #     )

        # plt.close()

        # # Reset confusion matrix for next epoch
        # self.confusion_matrix.reset()
        ...

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, Dict[str, Any]]]:

        optimizer_config = self.params.optimizer_config
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=optimizer_config.learning_rate,
            weight_decay=optimizer_config.weight_decay,
        )
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=optimizer_config.lr_max,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            pct_start=optimizer_config.pct_start,
            anneal_strategy=optimizer_config.anneal_strategy,
            cycle_momentum=True,
            base_momentum=optimizer_config.base_momentum,
            max_momentum=optimizer_config.max_momentum,
            div_factor=optimizer_config.div_factor,
            final_div_factor=optimizer_config.final_div_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 10,
                "monitor": "val_loss",
            },
        }

    def on_train_start(self) -> None:
        # Optional MLflow Integration
        self.logger.log_hyperparams(
            self.params.model_dump()
        )  # Save hyperparameters for reference
        # Further mlflow setup or configuration can be added here

    def on_train_end(self) -> None:
        # Optional ending process for mlflow or clean-up
        pass


if __name__ == "__main__":
    from torchsummary import summary

    params = ImgClassifierParams(model=ModelType.ALEXNET, num_classes=10)
    model = params.setup_target()
    summary(model, (3, 224, 224), device="cpu")
