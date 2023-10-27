from matplotlib_inline import backend_inline
import torch.nn.functional as F
import torch
import d2l
from IPython import display
import torchmetrics
import matplotlib.pyplot as plt


class Animator:
    """For plotting data in animation."""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_utils`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def show_img(img):
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.show()


def make_img():
    return torch.rand((1, 3, 512, 512))


def save_model(net, name):
    torch.save(net.state_dict(), f"{name}.pth")


def cross_entropy_loss(inputs, targets):
    loss = F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
    return loss.mean()


def dice_loss(predictions, labels, epsilon=1):
    #     num_classes = predictions.shape[1]
    predictions = F.softmax(predictions, dim=1)
    one_hot_labels = F.one_hot(labels, num_classes=2)
    dice_co_sum = 0
    for c in range(1, 2):

        # Flatten the predictions and labels for the current class
        predictions_c = predictions[:, c, :, :]
        pred_flat = predictions_c.reshape(-1)
        one_hot_labels_flat = one_hot_labels[:, :, :, c].reshape(-1)

        intersection = torch.sum(one_hot_labels_flat*pred_flat)
        union = torch.sum(pred_flat.pow(2) + one_hot_labels_flat.pow(2))
        # Compute the dice coefficient and add it to the total
        dice_coefficient = (2 * intersection + epsilon) / (union + epsilon)
        dice_co_sum += dice_coefficient
    dice_coefficient = dice_co_sum
    # Compute the average dice coefficient over all classes

    # Compute the dice loss as 1 - dice_coefficient
    dice_loss = 1 - dice_coefficient

    return dice_loss


def evaluate_iou(net, data_iter, device=None):
    net.eval()
    if not device:
        device = next(iter(net.parameters())).device
    iou_metric = torchmetrics.JaccardIndex(
        task='binary', num_classes=2).to('cuda')

    with torch.no_grad():
        ious = torch.tensor(0, dtype=torch.float32).to(device)
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            iou = iou_metric(net(X).argmax(dim=1), y)
    return iou


def accuracy(y_hat, y):
    """Compute the number of correct predictions.
    Defined in :numref:`sec_softmax_scratch`"""
    y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return cmp.type(y.dtype).sum()


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')


def show_label(label):
    plt.imshow(label.cpu().numpy())


def evaluate_accuracy(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_utils`"""

    net.eval()  # Set the model to evaluation mode
    if not device:
        device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
