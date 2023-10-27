from model.attentiondeeplab import AttentionDeeLabv3p
import numpy as np
import os
from utils.dataset import train_dataloader, test_dataloder
import torch
from torch import nn
from matplotlib import pyplot as plt
import torchvision
from PIL import Image
from torchvision import transforms
device = 'cuda'
model = AttentionDeeLabv3p().to(device)
model_path = './deeplab_attention.pth'
state_dict = torch.load('deeplab_attention.pth')
model.load_state_dict(state_dict)
# for i in test_dataloder:

#     img, label = i
#     img = img.to(device)
#     print(img.shape)
#     exit(0)
#     pred = model(img)
#     pred = pred.argmax(dim=1)
#     for i in range(pred.shape[0]):
#         fig, ax = plt.subplots(1, 1)
#         ax.imshow(pred[i].cpu().numpy())
#         plt.show()
#     break

im_path = 'newdata/frame_0000.jpg'
imgs = os.listdir('newdata')
for img in imgs:
    im_path = os.path.join('newdata', img)
    image = Image.open(im_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    resize = torchvision.transforms.Resize(256)
    image = resize(image)
    image_ = image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image = transform(image)

    image = torch.unsqueeze(image, 0).to(device)
    model.eval()

    pred = model(image).argmax(dim=1)
    # fig, ax = plt.subplots(1, 2, figsize=(40,40))
    # ax[0].imshow(pred[0].cpu().numpy())
    # ax[1].imshow(image_)
    # plt.show()
    fig, ax = plt.subplots()
    ax.imshow(image_)
    binary_mask = pred[0].cpu().numpy()
    color_mask = np.zeros(
        (binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    color_mask[binary_mask == 1] = [255, 0, 0]
    ax.imshow(color_mask, cmap='gray', alpha=0.5)
    plt.show()
    # break
