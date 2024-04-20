import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
from ..images_dataset import ImagesDataset
from ...utils.helper_funcs import get_images

class SWINSightTest:
    def __init__(self, model_path: str) -> None:
        self._transform = transforms.Compose([transforms.Resize((224,224)),transforms.PILToTensor()])
        self._test_loader = self._get_test_loader(self)

        self._data_len = len(os.listdir('SWINSight/dataset/test'))

        self._model = torch.load(model_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)

    def _get_test_loader(self):
        images, labels = get_images(dataset_path='SWINSight/dataset/test', transform=self._transform)
        data = ImagesDataset(images,labels)
        test_loader = DataLoader(data,batch_size=32)

        return test_loader
    
    def run_testing(self):
        running_loss = 0
        num_correct = 0
        criterion = torch.nn.BCEWithLogitsLoss()

        for image,label in tqdm(self._test_loader):
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
        test_loss = running_loss / len(self._data_len)
        test_acc = num_correct / len(self._data_len)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        return test_loss, test_acc
