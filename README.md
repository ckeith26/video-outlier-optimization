# Video Outlier Optimization

## [Website](https://video-outlier-optimization-web.onrender.com/)

## INSERT Motivation here

## Setup and Training

### Jupyter Notebooks [Independent Model Training](./training_files/)

You can train any of the models directly from the Jupyter notebooks. The trained models will be saved in the [models](./models/) directory.

### Pipeline [Model Training with Pipeline](./scripts/setup.py)

1. Download Docker: `pip install docker`
2. Run `sh ./scripts/run.sh` to build the Docker image and run the container.

**Or** manually run the following commands:
2. Build the Docker image: `docker build -t video_outlier_optimization -f ./container/Dockerfile .`
3. Run the Docker image: `docker run -p 8888:8888 video_outlier_optimization`

Troubleshooting: [Docker docs](https://docs.docker.com/get-docker/)

## Model Modification

### Training Parameters
- train_test_size: Number of training samples to use (e.g., 100k).
- train_test_split: The proportion of the dataset to include in the train split (e.g., 0.8).
- epochs: for training all models (e.g., 15).

### Data Parameters
- load_more_data: if True, the code will load a new dataset with the specified train_size.

### Configuration File 

[Modify config.ini](config.ini)

This code reads configuration values from an INI file named `config.ini` located in the parent directory (`./config.ini`). The configuration file should have the following structure:

```ini
[models]
model1 = ../training_files/base_model.ipynb
model2 = ../training_files/base_outlier_model.ipynb

[training]
train_size = 100000
train_test_split = 0.9
epochs = 15
batch_size = 128

[data]
load_more_data = False
train_dataset = ../data/train_data.pt
test_dataset = ../data/test_data.pt
train_subset_100k = ../data/train_subset_100k.pt
test_subset_20k = ../data/test_subset_20k.pt
train_subset_rest = ../data/train_subset_100k_rest.pt
test_subset_rest = ../data/test_subset_20k_rest.pt
```
