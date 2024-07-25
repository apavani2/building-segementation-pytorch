import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from models import get_model
from train import get_transforms
from typing import List, Tuple
from tqdm import tqdm


def load_model(
    model_path: str, num_classes: int = 1, device: torch.device = torch.device("cpu")
) -> torch.nn.Module:
    """
    Load the trained model from a checkpoint.

    Args:
        model_path (str): Path to the model checkpoint.
        num_classes (int): Number of output classes. Default is 1 for binary segmentation.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(
    image_path: str, transform: transforms.Compose, device: torch.device
) -> torch.Tensor:
    """
    Preprocess the input image.

    Args:
        image_path (str): Path to the input image.
        transform (transforms.Compose): Transformations to apply to the image.
        device (torch.device): Device to load the image tensor on.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def save_mask(mask: torch.Tensor, save_path: str) -> None:
    """
    Save the segmentation mask to a file.

    Args:
        mask (torch.Tensor): Segmentation mask tensor.
        save_path (str): Path to save the mask image.
    """
    mask_image = Image.fromarray((mask.squeeze().cpu().numpy() * 255).astype("uint8"))
    mask_image.save(save_path)


def run_inference(
    model: torch.nn.Module,
    image_path: str,
    transform: transforms.Compose,
    save_path: str,
    device: torch.device,
) -> None:
    """
    Run inference on a single image and save the result.

    Args:
        model (torch.nn.Module): The loaded model.
        image_path (str): Path to the input image.
        transform (transforms.Compose): Transformations to apply to the image.
        save_path (str): Path to save the resulting mask.
        device (torch.device): Device to run inference on.
    """
    image_tensor = preprocess_image(image_path, transform, device)
    with torch.no_grad():
        output = model(image_tensor)
        mask = (output > 0.5).float()
    save_mask(mask, save_path)


def main(
    image_dir: str, model_path: str, output_dir: str, use_gpu: bool = True
) -> None:
    """
    Main function to run inference on all images in a directory.

    Args:
        image_dir (str): Directory containing input images.
        model_path (str): Path to the model checkpoint.
        output_dir (str): Directory to save the output masks.
        use_gpu (bool): Whether to use GPU for inference if available.
    """
    os.makedirs(output_dir, exist_ok=True)
    transform = get_transforms(is_train=False)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(model_path, device=device)

    image_files = [
        f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    for image_name in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_name)
        save_path = os.path.join(
            output_dir, f"{os.path.splitext(image_name)[0]}_mask.png"
        )
        run_inference(model, image_path, transform, save_path, device)

    print(f"Processed {len(image_files)} images. Results saved in {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run inference on images using a trained segmentation model."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output masks.",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available.",
    )

    args = parser.parse_args()
    main(args.image_dir, args.model_path, args.output_dir, use_gpu=not args.use_cpu)
