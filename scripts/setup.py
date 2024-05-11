# type: ignore
import torch
import av
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import UCF101
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import traitlets
import requests
import patoolib
import papermill as pm

class setup():
    # choose models to train from scratch
    models = ["../training_files/base_model.ipynb", 
              "../training_files/base_outlier_model.ipynb"] # add more models here
    
    data_files = []
    train_samples = -1

    def __init__(self, models):
        models = models

    # execute
    pipeline()

    def pipeline():
        # load the data
        load_raw_data()
        load_data()
        # load the models
        for model in models:
            train_model(model)
            with open('../results/results.txt', 'r') as f:
                print(f.read())

    def train_model(model_path):
        # Load the notebook
        pm.execute_notebook(
            input_path = model_path,
            output_path = f"{model_path.split('.')[0]}_executed.ipynb",
            parameters = [self.data_files, train_samples]
        )

    def load_raw_data():
        # get videos
        url = "https://www.crcv.ucf.edu/datasets/human-actions/ucf101/UCF101.rar"
        response = requests.get(url, stream=True)

        with open("../data/UCF-101.rar", "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

        # Unrar the downloaded file
        patoolib.extract_archive("../data/UCF-101.rar", outdir="../data/UCF-101")

        # get labels
        url = "https://www.crcv.ucf.edu/wp-content/uploads/2019/06/Datasets_UCF101-VideoLevel.zip"
        response = requests.get(url, stream=True)

        with open("../data/ucfTrainTestlist.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

        # Unrar the downloaded file
        patoolib.extract_archive("../data/ucfTrainTestlist.zip", outdir="../data/ucfTrainTestlist")

    def load_data(train_size = 0.1, train_test_split = 0.8, batch_size = 128):
        frames_per_clip = 5
        step_between_clips = 1
        path='../data/ucfTrainTestlist'
        ucf_data_dir = "../data/UCF-101/"
        ucf_label_dir = path
        frames_per_clip = 5
        step_between_clips = 1

        def divide_by_255(x):
            return x / 255.

        def permute_channels(x):
            return x.permute(0, 3, 1, 2)

        def interpolate(x):
            return nn.functional.interpolate(x, (240, 320))

        tfs = transforms.Compose([
            transforms.Lambda(divide_by_255),
            transforms.Lambda(permute_channels),
            transforms.Lambda(interpolate),
        ])

        # load train test datasets
        train_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                            step_between_clips=step_between_clips, train=True, transform=tfs)

        test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                            step_between_clips=step_between_clips, train=False, transform=tfs)

        # Split the datasets
        train_size = train_test_split * train_size
        train_size_rest = train_test_split * (1 - train_size)
        test_size = (1 - train_test_split) * train_size
        test_size_rest = (1 - train_test_split) * (1 - train_size)

        train_set, test_set_org = torch.utils.data.random_split(train_dataset, [train_size, train_size_rest])
        test_set, test_set2 = torch.utils.data.random_split(test_dataset, [test_size, test_size_rest])

        self.train_samples = round(train_size/1000)

        with open('../data/train_dataset.pt', 'wb') as f:
            torch.save(train_dataset, f)

        # Save test_dataset
        with open('../data/test_dataset.pt', 'wb') as f:
            torch.save(test_dataset, f)

        with open(f'../data/train_dataset_{round(train_size/1000)}k.pt', 'wb') as f:
            torch.save(train_set, f)

        # Save test_dataset
        with open(f'../data/test_dataset_{round(test_size/1000)}k.pt', 'wb') as f:
            torch.save(test_set, f)

        with open(f'../data/train_dataset_{round(train_size_rest/1000)}k_rest.pt', 'wb') as f:
            torch.save(test_set_org, f)

        self.files.append(f'../data/train_dataset_{round(train_size/1000)}k.pt', 
                     f'../data/test_dataset_{round(test_size/1000)}k.pt', 
                     f'../data/train_dataset_{round(train_size_rest/1000)}k_rest.pt',
                     f'../data/test_dataset_{round(test_size_rest/1000)}k_rest.pt', 
                     '../data/train_dataset.pt', 
                     '../data/test_dataset.pt')

        # Save test_dataset
        with open(f'../data/test_dataset_{round(test_size_rest/1000)}k_rest.pt', 'wb') as f:
            torch.save(test_set2, f)

        return (train_loader_sub, test_loader_sub, train_loader_full, test_loader_full)
