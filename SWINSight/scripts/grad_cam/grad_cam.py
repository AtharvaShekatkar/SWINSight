import numpy as np
import cv2
import torch
import os
import argparse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image

def parse_args():
    parser = argparse.ArgumentParser(description='Process image using specified model.')
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to the input image file.'
    )

    parser.add_argument(
        '--model_name',
        type=str, 
        required=True,
        help='Name of the model to be used for processing.'
    )

    args = parser.parse_args()

    return args

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(3))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class SoftmaxOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return torch.softmax(model_output, dim=-1)

def main():
    args = parse_args()

    model = torch.load(f"SWINSight/model_history/{args.model_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _ = model.eval()
    model = model.to(device)
    target_layers = [model.layers[-1].blocks[-1].norm1]
    cam = GradCAM(model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform)


    rgb_img = cv2.imread(f"SWINSight/inputs/{args.image_path}", 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=None,
                    )
    grayscale_cam_img = grayscale_cam[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam_img)

    cv2.imwrite(f"SWINSight/outputs/{args.image_path}_gradcam.jpg", cam_image)
    # save cam_image


if __name__ == "__main__":
    main()