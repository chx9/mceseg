import torch
from datetime import datetime
import os
import os.path as osp
import os
from torch import nn
from torch.utils.data import DataLoader
from utils import read_config_file, accurate_cnt, overlay_mask_on_image,  set_mask_to_blue, set_masks_on_image
from utils.data import MceDataset
from model.attentiondeeplab import AttentionDeeLabv3p
from model.deeplab import DeeLabv3p
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
is_attention = True
parser = argparse.ArgumentParser(description='args')
parser.add_argument('--resume', action='store_true',
                    help='resume training from a saved checkpoint')

args = parser.parse_args()
resume = args.resume
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_path = './config.json'
config = read_config_file(config_path)

# train intervals
checkpoint_interval = config['train']['checkpoint_interval']
test_interval = config['test']['test_interval']
# hyper params
batch_size = config['data']['batch_size']
lr = config['train']['lr']
num_workers = config['data']['num_workers']
epochs = config['train']['epochs']
img_to_display = config['test']['img_to_display']
root_dir = config['data']['root_dir']
now = datetime.now()
writer = SummaryWriter()
if is_attention:
    model = AttentionDeeLabv3p().to(device)
    checkpoint_dir = 'checkpoints'
else:
    model = DeeLabv3p().to(device)
    checkpoint_dir = 'original_pth'
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
resume_epoch = 0
if resume:
    checkpoint_files = os.listdir('checkpoints')
    checkpoint_file = sorted(
        checkpoint_files, key=lambda x: int(x.split('_')[-1][:-4]))[-1]
    checkpoint = torch.load(osp.join('checkpoints', checkpoint_file))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    resume_epoch = checkpoint['epoch']

train_data = MceDataset(root_dir=root_dir, is_train=True)
test_data = MceDataset(root_dir=root_dir, is_train=False)
train_dataloader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, drop_last=True,)
iou_metric = torchmetrics.JaccardIndex(task='binary', num_classes=2).to('cuda')
dice_metric = torchmetrics.Dice().to('cuda')


def dice_coefficient(pred, target):
    smooth = 1.0  # 为了避免分母为零，添加一个平滑因子

    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def forward_step(model, images, labels, criterion, mode=''):
    if mode == 'test':
        with torch.no_grad():
            output = model(images)
    else:
        output = model(images)

    loss = criterion(output, labels)
    return loss, output


for epoch in range(resume_epoch, resume_epoch+epochs):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss, pred = forward_step(model, x, y, criterion, mode='train')
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print('='*20)
    writer.add_scalar('loss/train', epoch_loss/len(train_dataloader), epoch+1)
    print('train epoch: {}, loss:{}'.format(
        epoch+1, epoch_loss/len(train_dataloader)))
    print('='*20)
    # validation
    model.eval()
    if (epoch+1) % test_interval == 0:
        epoch_loss = 0
        correct_num = 0
        number_pixel = 0
        iou = 0
        dice = 0
        for i, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            loss, pred = forward_step(model, x, y, criterion, mode='test')
            correct_num += accurate_cnt(pred, y)
            number_pixel += y.numel()
            epoch_loss += loss.item()
            iou += iou_metric(pred.argmax(dim=1), y)
            # dice += dice_metric(pred.argmax(dim=1), y)
            dice += dice_coefficient(pred.argmax(dim=1), y)
            # Add image with masks to TensorBoard only for the first iteration
            if i == 0:
                # Assuming x has shape: (batch_size, channels, height, width) and channels is 3
                # Convert predictions to segmentation masks
                for j in range(min(img_to_display, x.size(0))):
                    pred_mask = pred.argmax(dim=1)

                    # Extract the original image, prediction mask, and ground truth mask
                    # Taking the first image in the batch
                    original_image = x[j].cpu()
                    # Taking the prediction mask for the first image
                    target_pred_mask = pred_mask[j].cpu().float()
                    # Taking the ground truth mask for the first image
                    target_gt_mask = y[j].cpu().float()

                    # Set the mask regions to colors in the original image
                    image_with_masks = set_masks_on_image(
                        original_image, target_pred_mask, target_gt_mask)

                    # Write image to TensorBoard
                    writer.add_image(
                        f'result_{j}', image_with_masks, global_step=epoch)

        epoch_loss /= len(test_dataloader)
        accurancy = correct_num / number_pixel
        iou = iou/len(test_dataloader)
        dice = dice/len(test_dataloader)
        print('*'*20)
        writer.add_scalar('loss/test', epoch_loss, epoch+1)
        writer.add_scalar('accurancy', accurancy, epoch+1)
        writer.add_scalar('iou', iou, epoch+1)
        writer.add_scalar('dice', dice, epoch+1)
        print('test epoch: {}, loss: {:.4f}, accurancy: {:.4f}, iou: {:.4f}, dice: {:.4f}'.format(
            epoch+1, epoch_loss, accurancy, iou, dice))
        print('*'*20)
        if (epoch+1) % checkpoint_interval == 0:
            # save model
            if not osp.exists(osp.join(checkpoint_dir)):
                os.mkdir(osp.join(checkpoint_dir))
            checkpoint_path = osp.join(
                checkpoint_dir, 'epoch_{}.pth'.format(epoch+1))
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
writer.close()
