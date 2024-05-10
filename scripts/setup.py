import torch
import av
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import UCF101
import nbformat
from nbconvert import HTMLExporter
import traitlets

class setup():

    models = []

    def __init__(self, models):
        models = models

    def load_model(model_path):
        # Load the notebook
        notebook = nbformat.read(model_path, as_version=4)

        # Configure the exporter
        html_exporter = HTMLExporter()
        html_exporter.template_file = 'basic' # Use the basic template

        # Execute the notebook
        ep = HTMLExporter(traitlets.config.Config())
        (body, resources) = ep.from_notebook_node(notebook)

    def load_raw_data():
        pass

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

        def custom_collate(batch):
            filtered_batch = []
            for video, _, label in batch:
                filtered_batch.append((video, label))
            return torch.utils.data.dataloader.default_collate(filtered_batch)

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

        # Save test_dataset
        with open(f'../data/test_dataset_{round(test_size_rest/1000)}k_rest.pt', 'wb') as f:
            torch.save(test_set2, f)

        # Load train_dataset
        with open('../data/train_dataset.pt', 'rb') as f:
            train_dataset = torch.load(f)

        # Load test_dataset
        with open('../data/test_dataset.pt', 'rb') as f:
            test_dataset = torch.load(f)

        # Load train_dataset_100k
        with open('../data/train_dataset_100k.pt', 'rb') as f:
            train_set = torch.load(f)

        # Load test_dataset_20k
        with open('../data/test_dataset_20k.pt', 'rb') as f:
            test_set = torch.load(f)

        # Load train_dataset_100k_rest
        with open('../data/train_dataset_100k_rest.pt', 'rb') as f:
            test_set_org = torch.load(f)

        # Load test_dataset_20k_rest
        with open('../data/test_dataset_20k_rest.pt', 'rb') as f:
            test_set2 = torch.load(f)
            
        train_loader_sub = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                collate_fn=custom_collate)
        test_loader_sub = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True,
                                                collate_fn=custom_collate)
        train_loader_full = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=custom_collate)
        test_loader_full = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                collate_fn=custom_collate)
        return (train_loader_sub, test_loader_sub, train_loader_full, test_loader_full)
