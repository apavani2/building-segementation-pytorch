import os
import argparse
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from utils import reformat_xyz, cleanup_invalid_geoms, generate_tiles
from typing import Tuple, List, Union


class BuildingDataset(Dataset):
    """
    A custom Dataset class for building segmentation data.

    Attributes:
        img_dir (Path): Directory containing image files.
        mask_dir (Path): Directory containing mask files.
        transform: Optional transform to be applied on a sample.
        images (List[str]): List of image file names.
        masks (List[str]): List of mask file names.
    """

    def __init__(self, img_dir: str, mask_dir: str, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0.5).float()

        return image, mask


def create_data_dir(data_dir: str) -> Tuple[Path, Path, Path, Path]:
    """
    Creates a data directory with separate folders for train and validation data,
    each containing images and masks subfolders.

    Args:
        data_dir (str): Path to the data directory.

    Returns:
        Tuple[Path, Path, Path, Path]: Paths to the train and val image and mask directories.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    # Create train directory and subdirectories
    train_dir = data_dir / "train"
    train_dir.mkdir(exist_ok=True)
    train_img_path = train_dir / "images-256"
    train_mask_path = train_dir / "masks-256"
    train_img_path.mkdir(exist_ok=True)
    train_mask_path.mkdir(exist_ok=True)

    # Create val directory and subdirectories
    val_dir = data_dir / "val"
    val_dir.mkdir(exist_ok=True)
    val_img_path = val_dir / "images-256"
    val_mask_path = val_dir / "masks-256"
    val_img_path.mkdir(exist_ok=True)
    val_mask_path.mkdir(exist_ok=True)

    return train_img_path, train_mask_path, val_img_path, val_mask_path


def load_geojson(url: str) -> gpd.GeoDataFrame:
    """
    Loads a GeoJSON file and removes empty rows.

    Args:
        url (str): URL or path to the GeoJSON file.

    Returns:
        gpd.GeoDataFrame: Loaded and cleaned GeoDataFrame.
    """
    label_df = gpd.read_file(url)
    label_df = label_df[label_df["geometry"].isna() != True]  # remove empty rows
    return label_df


def create_aoi_files(aoi_df: gpd.GeoDataFrame) -> None:
    """
    Creates separate GeoJSON files for training and validation areas of interest.

    Args:
        aoi_df (gpd.GeoDataFrame): GeoDataFrame containing areas of interest.
    """
    aoi_df[aoi_df["dataset"] == "trn"]["geometry"].to_file(
        "trn_aoi.geojson", driver="GeoJSON"
    )
    aoi_df[aoi_df["dataset"] == "val"]["geometry"].to_file(
        "val_aoi.geojson", driver="GeoJSON"
    )


def data_preprocess(args: argparse.Namespace) -> None:
    """
    Preprocesses the data by creating directories, loading and processing GeoJSON files,
    and generating image tiles.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Set up directories
    train_img_path, train_mask_path, val_img_path, val_mask_path = create_data_dir(
        args.data_dir
    )

    # Loading ground truth masks for the orthomosaic
    masks_df = load_geojson(args.building_masks)

    # Loading the training and validation aoi and splitting them to 2 separate geojson files
    aoi_df = gpd.read_file(args.train_val_split_aoi)
    create_aoi_files(aoi_df)

    # Breaking the train and val aoi to tiles
    os.system(
        f"cat trn_aoi.geojson | supermercado burn {args.map_zoom_level} | mercantile shapes | fio collect > trn_aoi_z{args.map_zoom_level}tiles.geojson"
    )
    os.system(
        f"cat val_aoi.geojson | supermercado burn {args.map_zoom_level} | mercantile shapes | fio collect > val_aoi_z{args.map_zoom_level}tiles.geojson"
    )

    # Loading the geojson tiles as train and val tiles
    trn_tiles = gpd.read_file(f"trn_aoi_z{args.map_zoom_level}tiles.geojson")
    val_tiles = gpd.read_file(f"val_aoi_z{args.map_zoom_level}tiles.geojson")
    trn_tiles["dataset"] = "trn"
    val_tiles["dataset"] = "val"

    # Remove overlapping tiles and create a geodataframe
    tiles_gdf = gpd.GeoDataFrame(
        pd.concat([trn_tiles, val_tiles], ignore_index=True), crs=trn_tiles.crs
    )
    tiles_gdf.drop_duplicates(subset=["id"], inplace=True)

    tiles_gdf = reformat_xyz(tiles_gdf)

    all_polys = masks_df.geometry
    all_polys = cleanup_invalid_geoms(all_polys)

    # Generate tiles for training data
    generate_tiles(
        tiles_gdf[tiles_gdf["dataset"] == "trn"],
        args.orthomosaic_tif,
        all_polys,
        args.map_tile_size,
        train_img_path,
        train_mask_path,
    )

    # Generate tiles for validation data
    generate_tiles(
        tiles_gdf[tiles_gdf["dataset"] == "val"],
        args.orthomosaic_tif,
        all_polys,
        args.map_tile_size,
        val_img_path,
        val_mask_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data Directory to store all the data",
    )
    parser.add_argument(
        "--orthomosaic_tif",
        type=str,
        default="tmp.tif",
        help="High resolution drone orthomosaic in tif format",
    )
    parser.add_argument(
        "--building_masks",
        type=str,
        default="grid_001.geojson",
        help="Building masks corresponding to the orthomosaic in geojson format",
    )
    parser.add_argument(
        "--map_zoom_level",
        type=int,
        default=19,
        help="Zoom Level of the orthomosaic maps",
    )
    parser.add_argument("--map_tile_size", type=int, default=256, help="Map tile size")

    parser.add_argument(
        "--train_val_split_aoi",
        type=str,
        default="aoi.geojson",
        help="Area of interest marked to create train and validation area of interets on orthomosaic",
    )

    args = parser.parse_args()

    data_preprocess(args)
