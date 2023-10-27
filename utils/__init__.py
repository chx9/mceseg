import json
import cv2
import torch
import torchmetrics
import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms


def read_config_file(file_path):
    with open(file_path) as f:
        json_data = json.load(f)
    return json_data


def accurate_cnt(y_hat, y):

    y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(y.dtype).sum()


def evaluate_accuracy(model, data_iter, device=None):

    model.eval()
    if not device:
        device = next(iter(model.parameters())).device
    number_pixel = 0
    correct_num = 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            correct_num += accurate_cnt(model(X))
            number_pixel += y.numel()
    return correct_num / number_pixel


def evaluate_iou(model, data_iter, device=None):
    model.eval()
    if not device:
        device = next(iter(model.parameters())).device
    iou_metric = torchmetrics.JaccardIndex(
        task='binary', num_classes=2).to('cuda')

    with torch.no_grad():
        ious = torch.tensor(0, dtype=torch.float32).to(device)
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            iou = iou_metric(model(X).argmax(dim=1), y)
    return iou


def overlay_mask_on_image(image, mask):
    # Convert the single channel mask into a 3 channel image with the blue channel set to the mask
    mask_overlay = torch.stack(
        [torch.zeros_like(mask), torch.zeros_like(mask), mask], dim=0)
    # Overlay the mask on the image
    overlay = image + mask_overlay
    # Clip values to the range [0, 1]
    overlay = torch.clamp(overlay, 0, 1)
    return overlay


def set_mask_to_blue(image, mask):
    # Create a blue mask with the same dimensions as the original image
    blue_mask = torch.stack(
        [torch.zeros_like(mask), torch.zeros_like(mask), mask], dim=0)
    # Set the regions identified by the mask to blue
    image_with_blue_mask = image.clone()
    image_with_blue_mask[:, mask > 0] = blue_mask[:, mask > 0]
    return image_with_blue_mask


def set_masks_on_image(image, pred_mask, gt_mask):
    # Create a blue mask for predictions and a green mask for ground truth
    blue_mask = torch.stack(
        [torch.zeros_like(pred_mask), torch.zeros_like(pred_mask), pred_mask], dim=0)
    green_mask = torch.stack(
        [torch.zeros_like(gt_mask), gt_mask, torch.zeros_like(gt_mask)], dim=0)

    # Set the regions identified by the masks to the respective colors
    image_with_masks = image.clone()
    image_with_masks[:, pred_mask > 0] = blue_mask[:, pred_mask > 0]
    image_with_masks[:, gt_mask > 0] = green_mask[:, gt_mask > 0]
    return image_with_masks
