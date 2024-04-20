import os
import numpy as np
import timm
from PIL import Image
import torchvision
from torch.utils.data import random_split
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import Linear

from ..images_dataset import ImagesDataset
from ...utils.helper_funcs import get_images
from typing import Literal

class SWINSightTrain:
    def __init__(self, pretrain: bool = True, size:Literal['base', 'large'] = 'large') -> None:
        self._transform = transforms.Compose([transforms.Resize((224,224)),transforms.PILToTensor()])
        self._train_loader, self._validation_loader = self._get_train_dataset_loaders()

        self._train_len = len(os.listdir('SWINSight/dataset/train'))
        self._val_len = len(os.listdir('SWINSight/dataset/validation'))

        self._model = timm.create_model(f'swin_{size}_patch4_window7_224.ms_in22k', pretrained=True, num_classes=1)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        
        self._pretrain = pretrain

        self._size = size


    def _get_train_dataset_loaders(self):
        train_images, train_labels = get_images(dataset_path='SWINSight/dataset/train', transform=self._transform)

        val_images, val_labels = get_images(dataset_path='SWINSight/dataset/validation', transform=self._transform)
        train_data = ImagesDataset(train_images,train_labels)
        val_test_data = ImagesDataset(val_images,val_labels)
        validation_dataset,test_dataset = random_split(val_test_data,[0.5,0.5],generator=torch.Generator().manual_seed(1))


        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        return train_loader, validation_loader

    
    def _pretrain_swin(self, epochs: int, lr: float):
        optimizer = torch.optim.AdamW(self._model.parameters(),lr = lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        best_model = ""

        patience = 5
        time_till_best = 0
        best_val_loss = np.inf
        if epochs == 1:
            save_str = f'SWINModel{self._size}_Pre'
        else:
            save_str = f'SWINModel{self._size}'
        
        for i in range(epochs):
            print(f"Starting Epoch Number: {i}")
            running_loss = 0.0
            num_correct = 0
            for image,label in tqdm(self._train_loader):
                label=torch.tensor(label,dtype=torch.float)
                image,label = image.to(self._device),label.to(self._device)
                optimizer.zero_grad()
                input = image.to(self._device)
                input = input/255
                outputs = self._model(input)
                loss = criterion(outputs, label.view(-1,1))
                running_loss += loss.item()
                loss.backward()
                prediction = torch.flatten(torch.round(torch.sigmoid(outputs)))
                num_correct += torch.sum(prediction == label)
                optimizer.step()
            train_loss = running_loss / self._train_len
            train_acc = num_correct / self._train_len
            num_correct = 0
            for image,label in tqdm(self._validation_loader):
                with torch.no_grad():
                    label=torch.tensor(label,dtype=torch.float)
                    image,label = image.to(self._device),label.to(self._device)
                    input = image.to(self._device)
                    input = input/255
                    outputs = self._model(input)
                    loss = criterion(outputs, label.view(-1,1))
                    running_loss += loss.item()
                    prediction = torch.flatten(torch.round(torch.sigmoid(outputs)))
                    num_correct += torch.sum(prediction == label)
            validation_loss = running_loss / self._val_len
            validation_acc = num_correct / self._val_len
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")
            print(f"Validation Loss: {validation_loss:.4f}")
            print(f"Validation Accuracy: {validation_acc:.4f}")
            torch.save(self._model,f"SWINSight/model_history/{save_str}_{i}")
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                time_till_best = 0
                best_model = f"SWINSight/model_history/{save_str}_{i}"
                continue
            time_till_best+=1
            if time_till_best>patience:
                print(f"Best_model:{best_model}")
                break

    def _freeze_swin_layers(self):
        for param in self._model.parameters():
            param.requires_grad = False
        for param in self._model.head.parameters():
            param.requires_grad = True

    def train_model(self):
        if self._pretrain:
            self._pretrain_swin(epochs=1, lr=1e-6)

        self._freeze_swin_layers()

        self._pretrain_swin(epochs=10, lr=5e-3)

        torch.save(self._model,f"SWINSight/model_history/SWINModelLarge_finetuned_hybrid")