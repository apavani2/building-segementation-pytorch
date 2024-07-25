import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import BuildingDataset
from models import get_model
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
import os
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from typing import Tuple, List, Optional


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.

    Loss = 1 - Dice_coefficient
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1.0
        input = torch.sigmoid(input)
        iflat = input.contiguous().view(-1).float()
        tflat = target.contiguous().view(-1).float()
        # overlap between prediction and ground truth
        intersection = (iflat * tflat).sum()
        loss = 1 - ((2.0 * intersection + smooth) / ((iflat + tflat).sum() + smooth))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary segmentation tasks.

    pt: model's estimated probability for true class
    alpha: balancing factor for class imbalance
    gamma: is a focusing parameter that adjusts the rate at which easy examples are down-weighted.

    """

    def __init__(self, alpha: float = 1, gamma: float = 2, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        pt = torch.exp(-BCE_loss)

        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            return F_loss


class NormalizeImage:
    """
    Custom normalization transform for images.
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 3:  # Only normalize if it's an RGB image
            return transforms.functional.normalize(x, self.mean, self.std)
        return x


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Get the appropriate transforms for training or validation.
    """
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class SegmentationMetrics:
    """
    Class to compute and store segmentation metrics.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.iou = BinaryJaccardIndex().to(device)
        self.dice = BinaryF1Score().to(device)
        self.pixel_accuracy = Accuracy(task="binary").to(device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = (preds > 0.5).float()
        self.iou.update(preds, targets)
        self.dice.update(preds, targets)
        self.pixel_accuracy.update(preds, targets)

    def compute(self) -> Tuple[float, float, float]:
        iou = self.iou.compute().item()
        dice = self.dice.compute().item()
        pixel_accuracy = self.pixel_accuracy.compute().item()
        return iou, dice, pixel_accuracy

    def reset(self) -> None:
        self.iou.reset()
        self.dice.reset()
        self.pixel_accuracy.reset()


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to set up and run the training process.
    """
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb
    wandb.init(
        project=cfg.wandb.project_name, config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Get transforms
    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)

    # Set up datasets and dataloaders
    train_dataset = BuildingDataset(
        cfg.data.train_images, cfg.data.train_masks, transform=train_transform
    )
    val_dataset = BuildingDataset(
        cfg.data.val_images, cfg.data.val_masks, transform=val_transform
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False
    )

    # Initialize model, loss, and optimizer
    model = get_model(num_classes=1)
    dice_loss = DiceLoss()
    focal_loss = FocalLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Initialize the scheduler
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    # Train the model
    train(
        model,
        train_loader,
        val_loader,
        dice_loss,
        focal_loss,
        optimizer,
        scheduler,
        cfg.training.num_epochs,
        cfg.training.save_dir,
    )

    wandb.finish()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dice_loss: nn.Module,
    focal_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    save_dir: str,
) -> None:
    """
    Training loop for the segmentation model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = SegmentationMetrics(device=device)

    best_val_dice = 0.0

    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        train_loss = 0
        metrics.reset()

        for images, masks in tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False
        ):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            d_loss = dice_loss(outputs, masks)
            f_loss = focal_loss(outputs, masks)
            loss = d_loss + f_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            metrics.update(outputs, masks)

        avg_train_loss = train_loss / len(train_loader)
        train_iou, train_dice, train_pixel_accuracy = metrics.compute()

        current_lr = optimizer.param_groups[0]["lr"]

        # Validation
        model.eval()
        val_loss = 0
        metrics.reset()

        with torch.no_grad():
            for images, masks in tqdm(
                val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", leave=False
            ):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                d_loss = dice_loss(outputs, masks)
                f_loss = focal_loss(outputs, masks)
                val_loss += (d_loss + f_loss).item()
                metrics.update(outputs, masks)

        avg_val_loss = val_loss / len(val_loader)
        val_iou, val_dice, val_pixel_accuracy = metrics.compute()

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(
            f"Train IoU: {train_iou:.4f}, Train Dice: {train_dice:.4f}, Train Pixel Accuracy: {train_pixel_accuracy:.4f}"
        )
        print(
            f"Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}, Val Pixel Accuracy: {val_pixel_accuracy:.4f}\n"
        )
        print(f"Learning Rate: {current_lr:.6f}\n")

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_iou": train_iou,
                "train_dice": train_dice,
                "train_pixel_accuracy": train_pixel_accuracy,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "val_pixel_accuracy": val_pixel_accuracy,
                "learning_rate": current_lr,
            }
        )

        scheduler.step()

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        # Save best model based on validation IoU
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

    print("Training completed.")


if __name__ == "__main__":
    main()
