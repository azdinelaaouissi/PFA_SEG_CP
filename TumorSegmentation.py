# Define the TumorSegmentation class as a LightningModule
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import imgaug.augmenters as iaa

import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from celluloid import Camera
from IPython.display import HTML



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Define a sequence of two convolutional layers followed by ReLU activation.
        self.step = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )

    def forward(self, X):
        # Pass the input tensor through the sequence of layers.
        return self.step(X)


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Define the encoder part of the UNet with four DoubleConvBlock layers.
        self.layer1 = DoubleConvBlock(1, 64)
        self.layer2 = DoubleConvBlock(64, 128)
        self.layer3 = DoubleConvBlock(128, 256)
        self.layer4 = DoubleConvBlock(256, 512)

        # Define the decoder part of the UNet with three DoubleConvBlock layers.
        # The input channels for each layer are the concatenation of the output of the previous decoder layer and the corresponding encoder layer.
        self.layer5 = DoubleConvBlock(512 + 256, 256)
        self.layer6 = DoubleConvBlock(256 + 128, 128)
        self.layer7 = DoubleConvBlock(128 + 64, 64)

        # Final 1x1 convolutional layer to produce the output segmentation mask.
        self.layer8 = torch.nn.Conv2d(64, 1, 1)

        # Max pooling layer with kernel size 2 and stride 2 to downsample the feature maps.
        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder part of the UNet:
        x1 = self.layer1(x)
        x1_m = self.maxpool(x1)

        x2 = self.layer2(x1_m)
        x2_m = self.maxpool(x2)

        x3 = self.layer3(x2_m)
        x3_m = self.maxpool(x3)

        x4 = self.layer4(x3_m)

        # Decoder part of the UNet with upsampling using bilinear interpolation (Upsample) and concatenation of skip connections.
        x5 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x4)
        x5 = torch.cat([x5, x3], dim=1)  # Skip-connection
        x5 = self.layer5(x5)

        x6 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x5)
        x6 = torch.cat([x6, x2], dim=1)  # Skip-connection
        x6 = self.layer6(x6)

        x7 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)  # Skip-connection
        x7 = self.layer7(x7)

        # Final 1x1 convolutional layer to produce the output segmentation mask.
        return self.layer8(x7)

class TumorSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Create an instance of the UNet model
        self.model = UNet()

        # Define the Adam optimizer for model parameters with a learning rate of 1e-4
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # Define the loss function for binary classification with logits
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    # Define the forward pass of the model
    def forward(self, data):
        pred = self.model(data)
        return pred

    # Define the training step for the LightningModule
    def training_step(self, batch, batch_idx):
        # Unpack the batch into CT scan images and their corresponding masks
        ct_scan, mask = batch
        mask = mask.float()

        # Perform the forward pass to get the model predictions
        pred = self(ct_scan)

        # Calculate the binary cross-entropy loss between predictions and masks
        loss = self.loss_fn(pred, mask)

        # Log the loss and images periodically during training
        self.log("Train Loss", loss)
        if batch_idx % 50 == 0:
            self.log_images(ct_scan.cpu(), pred.cpu(), mask.cpu(), "Train")

        return loss

    # Define the validation step for the LightningModule
    def validation_step(self, batch, batch_idx):
        # Unpack the batch into CT scan images and their corresponding masks
        ct_scan, mask = batch
        mask = mask.float()

        # Perform the forward pass to get the model predictions
        pred = self(ct_scan)

        # Calculate the binary cross-entropy loss between predictions and masks
        loss = self.loss_fn(pred, mask)

        # Log the loss and images periodically during validation
        self.log("Val Loss", loss)
        if batch_idx % 50 == 0:
            self.log_images(ct_scan.cpu(), pred.cpu(), mask.cpu(), "Val")

        return loss

    # Define a method to log CT scan images, actual masks, and predicted masks
    def log_images(self, ct_scan, pred, mask, name):
        results = []

        # Threshold the predicted masks to obtain binary masks
        pred = pred > 0.5

        # Create a figure with two subplots to display actual and predicted masks
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(ct_scan[0][0], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][0] == 0, mask[0][0])
        axis[0].imshow(mask_, alpha=0.6)
        axis[0].set_title("Actual")

        axis[1].imshow(ct_scan[0][0], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][0] == 0, mask[0][0])
        axis[1].imshow(mask_, alpha=0.6)
        axis[1].set_title("Predicted")

        # Add the figure to the experiment's logger for visualization
        self.logger.experiment.add_figure(f"{name} Actual vs Prediction", fig, self.global_step)

    # Define a method to configure the optimizer
    def configure_optimizers(self):
        return [self.optimizer]
