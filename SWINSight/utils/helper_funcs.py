from PIL import Image
from torchvision import transforms
import os
from typing import List, Tuple
from tqdm import tqdm

def get_images(dataset_path: str, transform: transforms.Compose) -> Tuple[List, List]:
    images = []
    labels = []

    for image_path in tqdm(os.listdir(f"{dataset_path}/real_images")):
        images.append(transform(Image.open(f"{dataset_path}/real_images/{image_path}").convert('RGB')))
        labels.append(0)
    for image_path in tqdm(os.listdir(f"{dataset_path}/fake_images")):
        images.append(transform(Image.open(f"{dataset_path}/fake_images/{image_path}").convert('RGB')))
        labels.append(1)

    return images, labels