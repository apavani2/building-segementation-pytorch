import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import box, Polygon
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from rasterio.features import rasterize
import skimage.io
from tqdm import tqdm
from typing import List, Tuple, Union
from rio_tiler.io import COGReader


def reformat_xyz(tile_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Reformats the 'id' column of a GeoDataFrame to extract x, y, z coordinates.

    Args:
        tile_gdf (gpd.GeoDataFrame): Input GeoDataFrame with 'id' column.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with reformatted 'xyz' column.
    """
    tile_gdf["xyz"] = tile_gdf.id.apply(
        lambda x: x.lstrip("(,)").rstrip("(,)").split(",")
    )
    tile_gdf["xyz"] = [[int(q) for q in p] for p in tile_gdf["xyz"]]
    return tile_gdf


def explode(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Explodes multi-part geometries into single-part geometries.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame with multi-part geometries.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with single-part geometries.
    """
    gs = gdf.explode(index_parts=True)
    gdf_out = gs.reset_index(level=1, drop=True).rename(columns={0: "geometry"})
    gdf_out.crs = gdf.crs
    return gdf_out


def cleanup_invalid_geoms(all_polys: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Cleans up invalid geometries by applying a buffer and merging.

    Args:
        all_polys (gpd.GeoSeries): Input GeoSeries of polygons.

    Returns:
        gpd.GeoSeries: Cleaned up GeoSeries of polygons.
    """
    all_polys_merged = gpd.GeoDataFrame()
    all_polys_merged["geometry"] = gpd.GeoSeries(
        unary_union([p.buffer(0) for p in all_polys])
    )

    gdf_out = explode(all_polys_merged)
    gdf_out = gdf_out.reset_index()
    all_polys = gdf_out["geometry"]
    return all_polys


def save_tile_img(
    tif_url: str,
    xyz: Tuple[int, int, int],
    tile_size: int,
    save_path: str = "",
    prefix: str = "",
) -> None:
    """
    Saves a tile image from a COG file.

    Args:
        tif_url (str): URL of the COG file.
        xyz (Tuple[int, int, int]): x, y, z coordinates of the tile.
        tile_size (int): Size of the tile.
        save_path (str, optional): Path to save the image. Defaults to "".
        prefix (str, optional): Prefix for the saved image filename. Defaults to "".
    """
    x, y, z = xyz
    with COGReader(tif_url) as cog:
        tile, mask = cog.tile(x, y, z, tilesize=tile_size)
    skimage.io.imsave(
        f"{save_path}/{prefix}{z}_{x}_{y}.png",
        np.moveaxis(tile, 0, 2),
        check_contrast=False,
    )


def save_tile_mask(
    labels_poly: gpd.GeoSeries,
    tile_poly: Polygon,
    xyz: Tuple[int, int, int],
    tile_size: int,
    save_path: str = "",
    prefix: str = "",
) -> None:
    """
    Saves a tile mask from a set of polygons.

    Args:
        labels_poly (gpd.GeoSeries): GeoSeries of label polygons.
        tile_poly (Polygon): Polygon representing the tile area.
        xyz (Tuple[int, int, int]): x, y, z coordinates of the tile.
        tile_size (int): Size of the tile.
        save_path (str, optional): Path to save the mask. Defaults to "".
        prefix (str, optional): Prefix for the saved mask filename. Defaults to "".
    """
    x, y, z = xyz
    tfm = from_bounds(*tile_poly.bounds, tile_size, tile_size)

    # Create a bounding box for the tile
    tile_box = box(*tile_poly.bounds)

    # Crop polygons to the tile
    cropped_polys = [
        poly.intersection(tile_box) for poly in labels_poly if poly.intersects(tile_box)
    ]

    if len(cropped_polys) > 0:
        print(f"Number of cropped polygons: {len(cropped_polys)}")

    if len(cropped_polys) == 0:
        inverted_mask = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    else:
        # Create masks
        footprint_mask = rasterize(
            [(geom, 1) for geom in cropped_polys],
            out_shape=(tile_size, tile_size),
            transform=tfm,
            fill=0,
            dtype=np.uint8,
        )

        inverted_mask = 1 - footprint_mask

    plt.imsave(
        f"{save_path}/{prefix}{z}_{x}_{y}_mask.png", inverted_mask, cmap="binary"
    )


def generate_tiles(
    tiles_gdf: gpd.GeoDataFrame,
    tif_url: str,
    all_polys: gpd.GeoSeries,
    tile_size: int,
    img_path: str,
    mask_path: str,
) -> None:
    """
    Generates image and mask tiles from a GeoDataFrame of tile information.

    Args:
        tiles_gdf (gpd.GeoDataFrame): GeoDataFrame containing tile information.
        tif_url (str): URL of the COG file.
        all_polys (gpd.GeoSeries): GeoSeries of all polygons.
        tile_size (int): Size of the tiles.
        img_path (str): Path to save image tiles.
        mask_path (str): Path to save mask tiles.
    """

    print(f"Number of polygons in all_polys: {len(all_polys)}")

    for idx, tile in tqdm(tiles_gdf.iterrows(), total=len(tiles_gdf)):
        try:
            dataset = tile["dataset"]
            save_tile_img(
                tif_url,
                tile["xyz"],
                tile_size,
                save_path=img_path,
                prefix=f"znz001{dataset}_",
            )
            tile_poly = tile["geometry"]
            save_tile_mask(
                all_polys,
                tile_poly,
                tile["xyz"],
                tile_size,
                save_path=mask_path,
                prefix=f"znz001{dataset}_",
            )
        except Exception as e:
            print(f"Error processing tile {tile['xyz']}: {e}")
