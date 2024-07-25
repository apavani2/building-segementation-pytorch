# Building Segmentation in Drone Imagery with Pytorch

This project implements a building segmentation model using the DeepLabV3+ architecture. It includes scripts for data preparation, training, and inference.

## Setup
1. Create a conda environment

```conda create -n drone-seg python==3.9```

2. Install the required dependencies:

```pip install -r requirements.txt```


## Data Preparation

1. Organize your data in the following structure:
```
data/
├── train
│ ├── images-256/
│ └── masks-256/
└── val/
│ ├── images-256/
│ └── masks-256/
```


2. Run the data preparation script:

```python dataset.py --data_dir data --orthomosaic_tif <path_to_orthomosaic.tif> --building_masks <path_to_building_masks.geojson> --train_val_split_aoi <path_to_aoi.geojson> --map_zoom_level 19 --map_tile_size 256```


This script will process the orthomosaic, create image tiles, and generate corresponding masks.

## Training

1. Configure the training parameters in `conf/config.yaml`.

2. Start the training process:

```python train.py```

The script will use the configuration from `conf/config.yaml`. You can override parameters using command-line arguments, e.g.:

```python train.py training.batch_size=16 training.num_epochs=50```

3. Monitor the training progress using Weights & Biases (wandb).

## Inference

To run inference on a set of images:

1. Ensure you have a trained model checkpoint.

2. Run the inference script:

```python inference.py --image_dir <path/to/test/images> --model_path <path/to/best_model.pth> --output_dir <path/to/output>```

This will process all images in the specified directory and save the resulting masks in the output directory.

To run inference on **CPU** (even if GPU is available), add the `--use_cpu` flag:

```python inference.py --image_dir <path/to/test/images> --model_path <path/to/best_model.pth> --output_dir <path/to/output> --use_cpu```


## Project Structure

- `dataset.py`: Script for data preparation and preprocessing.
- `train.py`: Main training script.
- `inference.py`: Script for running inference on new images.
- `models.py`: Contains the model architecture definition.
- `utils.py` : Contains utility function for data preparation step 
- `conf/config.yaml`: Configuration file for training parameters.

## Additional Notes

- Make sure to adjust the paths in the commands according to your directory structure.
- The training script uses Weights & Biases for logging. Make sure to set up your wandb account.\

## Model Architecture

The project uses DeepLabV3+ with a ResNet-101 backbone for semantic segmentation. The model is defined in `models.py`.

## Loss Functions

Two loss functions are used in combination:
1. Dice Loss: Measures the overlap between the predicted and ground truth masks.
2. Focal Loss: Addresses class imbalance in binary segmentation tasks.

## Learning Rate Scheduler

A StepLR scheduler is used to adjust the learning rate during training. It reduces the learning rate by a factor of 0.5 every 4 epochs.

## Metrics

The following metrics are tracked during training and validation:
- IoU (Intersection over Union)
- Dice Coefficient (F1 Score)
- Pixel Accuracy

## Data Augmentation

Data augmentation is applied during training, including random horizontal and vertical flips. You can modify the `get_transforms` function in `train.py` to add more augmentations if needed.

## Customization

- To modify the model architecture, update the `get_model` function in `models.py`.
- To change the learning rate schedule, adjust the `scheduler` initialization in `train.py`.
- To add or modify data augmentations, update the `get_transforms` function in `train.py`.

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are correctly installed.
2. Check that the data is organized in the correct directory structure.
3. Verify that the paths in the configuration file are correct.
4. For GPU-related issues, make sure you have the correct CUDA version installed for your PyTorch version.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
