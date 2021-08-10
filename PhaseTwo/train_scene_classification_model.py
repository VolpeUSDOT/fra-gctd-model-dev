##################################################
## Script to train a computer vision network to perform scene classification. 
## This script fine-tunes a pre-trained computer vision network
##################################################
## MIT License
##################################################
## Author: Robert Rittmuller
## Copyright: Copyright 2021, Volpe National Transportation Systems Center
## Credits: Robert Rittmuller
## License: MIT
## Version: 0.1.1
## Maintainer: Robert Rittmuller
## Email: robert.rittmuller@dot.gov
## Status: Active Development
##################################################
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
from pathlib import Path
from collections import Counter
from torchvision.transforms.transforms import Resize
from scene_training_utils import *

# training parameters
batch_size = 32
num_epochs = 26
learning_rate = 0.002
learning_momentum = 0.9
num_workers = 12
feature_extract = True

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    print("Device in use: ",device)

    # NOTE: project directory structure is as follows:
    # ./example_dataset/
    #       train/
    #       val/
    # ./models
    # ./report

    # location of the training data
    data_dir = "/mnt/ml_data/FRA/Phase2/TrainingData/"
    dataset_name = "grade_v1"

    # where to put any visualizations
    report_path = Path("PhaseTwo/report")
    if not os.path.isdir(report_path):
        os.mkdir(report_path)

    # Where to save the class labels
    label_file = "PhaseTwo/models/labels.txt"

    # where to put the trainied models
    model_path = Path("PhaseTwo/models")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # Types of models =  [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_names = ['resnet','vgg','inception','squeezenet','densenet']
    # model_names = ['squeezenet']

    results = []

    for model_name in model_names:

        print("Initializing Datasets and Dataloaders...")

        model_ft, input_size = initialize_model(models, model_name, 1, feature_extract, use_pretrained=True)

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1),
                transforms.RandomRotation(1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

        dataset = datasets.ImageFolder(os.path.join(data_dir, dataset_name), transform=data_transforms['train'])
        image_datasets = train_val_dataset(dataset)
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers) for x in ['train', 'val']}
        labels = image_datasets['train'].dataset.classes
        num_classes = len(labels)

        model_ft, input_size = initialize_model(models, model_name, num_classes, feature_extract, use_pretrained=True)
        
        model_ft = model_ft.to(device)

        params_to_update = model_ft.parameters()

        training_class_counts = dict(Counter(sample_tup[1] for sample_tup in image_datasets['train']))
        training_class_counts = dict(sorted(training_class_counts.items()))
        print('Training label counts\t',training_class_counts)

        validation_class_counts = dict(Counter(sample_tup[1] for sample_tup in image_datasets['val']))
        validation_class_counts = dict(sorted(validation_class_counts.items()))
        print('Validation label counts\t',validation_class_counts)

        classNames = dataset.classes
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=learning_momentum)

        criterion = nn.CrossEntropyLoss()

        model_ft, hist, best_acc = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_classes, training_class_counts, validation_class_counts, labels, num_epochs=num_epochs, is_inception=(model_name=="inception"))

        ohist = []

        ohist = [float(h) for h in hist]

        results.append(ohist)

        plt.title("Validation Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Validation Accuracy")
        plt.plot(range(1,num_epochs+1),ohist,label='{:.0%}'.format(best_acc) + " Pretrained " + model_name)
        plt.ylim((0,1.))
        plt.xticks(np.arange(1, num_epochs+1, 0.5))
        plt.legend()
        plt.savefig(report_path / 'model-comparison.jpg')

        save_labels(labels, label_file)
        torch.save(model_ft, model_path / Path('saved_model_' + model_name + '.pt'))