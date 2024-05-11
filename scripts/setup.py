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
import configparser
import subprocess
import sys

class setup:
    def __init__(self):
        self.config = self.read_config('config.ini')
        self.models = [self.config.get('models', 'model1'), self.config.get('models', 'model2')]
        self.train_size = self.config.getfloat('training', 'train_size')
        self.train_test_split = self.config.getfloat('training', 'train_test_split')
        self.epochs = self.config.getint('training', 'epochs')
        self.load_more_data = self.config.getboolean('data', 'load_more_data')

    def read_config(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config
    
    def install_kernel(self):
        subprocess.check_call([sys.executable, "-m", "ipykernel", "install", "--user", "--name=my_kernel"])
        
    def pipeline(self):
        # if you want to load more data
        if self.load_more_data:
            self.load_raw_data()
            self.load_more_data()
        # load the models
        self.install_kernel()
        for model in self.models:
            
            self.train_model(model)
            with open('../results/results.txt', 'r') as f:
                print(f.read())

    def train_model(self, model_path):
        # Load the notebook
        pm.execute_notebook(
            input_path = model_path,
            output_path = f"{model_path.split('.')[0]}_executed.ipynb",
            kernel = 'my_kernel'
            # parameters = [self.data_files, train_samples]
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

    def load_more_data(self):
        frames_per_clip = 5
        step_between_clips = 1
        path='../data/ucfTrainTestlist'
        ucf_data_dir = "../data/UCF-101/"
        ucf_label_dir = path

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

        if len(train_dataset) < self.train_size:
            raise ValueError('train_size is larger than the dataset size')
        
        if self.train_test_split > 1 or self.train_test_split < 0:
            raise ValueError('train_test_split must be between 0 and 1')
        
        # Split the datasets
        train_size_rest = len(train_dataset) - self.train_size
        test_size = round((self.train_size / (self.train_test_split)) * (1 - self.train_test_split))
        test_size_rest = len(test_dataset) - test_size

        train_set, test_set_org = torch.utils.data.random_split(train_dataset, [self.train_size, train_size_rest])
        test_set, test_set2 = torch.utils.data.random_split(test_dataset, [test_size, test_size_rest])

        with open('../data/train_dataset.pt', 'wb') as f:
            torch.save(train_dataset, f)

        # Save test_dataset
        with open('../data/test_dataset.pt', 'wb') as f:
            torch.save(test_dataset, f)

        with open(f'../data/train_dataset_{round(train_test_size/1000)}k.pt', 'wb') as f:
            torch.save(train_set, f)

        # Save test_dataset
        with open(f'../data/test_dataset_{round(test_size/1000)}k.pt', 'wb') as f:
            torch.save(test_set, f)

        with open(f'../data/train_dataset_{round(train_test_samples_rest/1000)}k_rest.pt', 'wb') as f:
            torch.save(test_set_org, f)

        # self.data_files.append(
        #             f'../data/test_dataset_{round(test_size_rest/1000)}k_rest.pt', 
        #             f'../data/test_dataset_{round(test_size/1000)}k.pt', 
        #             '../data/test_dataset.pt'    
        #             f'../data/train_dataset_{round(train_size_rest/1000)}k_rest.pt',
        #             f'../data/train_dataset_{round(self.train_size/1000)}k.pt', 
        #              '../data/train_dataset.pt', 
        #              )
        config['data'][f'train_dataset'] = '../data/train_dataset.pt'
        config['data'][f'test_dataset'] = '../data/test_dataset.pt'
        config['data'][f'train_subset_{round(train_size/1000)}k'] = f'../data/train_subset_{round(train_size/1000)}k.pt'        
        config['data'][f'train_subset_{round(train_size_rest/1000)}k_rest'] = f'../data/train_subset_{round(train_size_rest/1000)}k_rest.pt'
        config['data'][f'test_subset_{round(test_size/1000)}k'] = f'../data/test_subset_{round(test_size/1000)}k.pt'
        config['data'][f'test_size_{round(test_size_rest/1000)}k_rest'] = f'../data/test_subset_{round(test_size_rest/1000)}k_rest.pt'

        with open('config.ini', 'w') as configfile:
            config.write(configfile)

        # Save test_dataset
        with open(f'../data/test_dataset_{round(test_size_rest/1000)}k_rest.pt', 'wb') as f:
            torch.save(test_set2, f)

        return train_loader_sub, test_loader_sub, train_loader_full, test_loader_full

# start the pipeline
execute = setup()
execute.pipeline()
