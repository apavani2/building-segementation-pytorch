import os
import pandas as pd
import geopandas as gpd
from dataset import create_data_dir, load_geojson, create_aoi_files
from utils import reformat_xyz, cleanup_invalid_geoms, generate_tiles
import argparse

def data_preprocess(args):
    # Set up directories
    img_path, mask_path = create_data_dir(args.data_dir)

    # Loading ground truth masks for the orthomosaic     
    masks_df = load_geojson(args.building_masks)

    # Loading the training and validation aoi and splitting them to 2 separate geojson files 
    aoi_df = gpd.read_file(args.train_val_split_aoi)
    create_aoi_files(aoi_df)

    #Breaking the train and val aoi to tiles
    os.system(f'cat trn_aoi.geojson | supermercado burn {args.map_zoom_level} | mercantile shapes | fio collect > trn_aoi_z{args.map_zoom_level}tiles.geojson')
    os.system(f'cat val_aoi.geojson | supermercado burn {args.map_zoom_level} | mercantile shapes | fio collect > val_aoi_z{args.map_zoom_level}tiles.geojson')

    # Loading the geojson tiles as train and val tiles
    trn_tiles = gpd.read_file(f'trn_aoi_z{args.map_zoom_level}tiles.geojson')
    val_tiles = gpd.read_file(f'val_aoi_z{args.map_zoom_level}tiles.geojson')
    trn_tiles['dataset'] = 'trn'
    val_tiles['dataset'] = 'val'

    #Remove overlapping tiles and create a geodataframe
    tiles_gdf = gpd.GeoDataFrame(pd.concat([trn_tiles, val_tiles], ignore_index=True), crs=trn_tiles.crs)
    tiles_gdf.drop_duplicates(subset=['id'], inplace=True)

    tiles_gdf = reformat_xyz(tiles_gdf)

    all_polys = masks_df.geometry
    all_polys = cleanup_invalid_geoms(all_polys)

    generate_tiles(tiles_gdf, args.orthomosiac_tif, all_polys, args.map_tile_size, img_path, mask_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Data Directory to store all the data')
    parser.add_argument('--orthomosaic_tif', type=str, default='tmp.tif', help='High resolution drone orthomosaic in tif format')
    parser.add_argument('--building_masks', type=str, default='grid_001.geojson', help='Building masks corresponding to the orthomosaic in geojson format')
    parser.add_argument('--map_zoom_level', type=int, default=19, help='Zoom Level of the orthomosaic maps')
    parser.add_argument('--map_tile_size', type=int, default=256, help='Map tile size')
    parser.add_argument('--train_val_split_aoi', type=str, default='aoi.geojson', help='Area of interest marked to create train and validation area of interets on orthomosaic')

    args = parser.parse_args()

    data_preprocess(args)