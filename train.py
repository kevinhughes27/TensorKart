#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from torch.utils.data import DataLoader, Dataset

from utils import Sample

# Global variables
OUT_SHAPE = 5
INPUT_SHAPE = (Sample.IMG_D, Sample.IMG_H, Sample.IMG_W)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Model(nn.Module):
    """Pretrained backbone with a linear head for the joystick output"""

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=pretrained, num_classes=OUT_SHAPE)
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.backbone(x)


def create_model(pretrained=True):
    return Model(pretrained)


def output_activation(y):
    """tanh for the joystick axes, sigmoid for the buttons"""
    return torch.cat([torch.tanh(y[..., :2]), torch.sigmoid(y[..., 2:])], dim=-1)


def customized_loss(y_pred, y_true):
    # MSE on the joystick axes, cross entropy on the buttons
    axes = F.mse_loss(torch.tanh(y_pred[:, :2]), y_true[:, :2])
    buttons = F.binary_cross_entropy_with_logits(y_pred[:, 2:], y_true[:, 2:])
    return axes + buttons


class DrivingDataset(Dataset):
    """Raw uint8 frames with augmentation applied on the fly.

    Augmentation is limited to a horizontal flip (with the steering
    axis negated) and brightness/contrast jitter. No translation or
    cropping - the steering label is tied to the camera geometry.
    The flip also balances the data: the recorded tracks turn left
    far more often than right.
    """

    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X) # uint8 (N, H, W, C)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].float() / 255.0
        y = self.y[i].clone()
        if self.augment:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[1]) # mirror left/right
                y[0] = -y[0]                # steering negates
            if torch.rand(()) < 0.8:
                brightness = 1.0 + 0.4 * (torch.rand(()) - 0.5)
                contrast = 1.0 + 0.4 * (torch.rand(()) - 0.5)
                x = ((x - 0.5) * contrast + 0.5) * brightness
                x = x.clamp(0.0, 1.0)
        return x.permute(2, 0, 1), y


if __name__ == '__main__':
    # Load Training Data
    x_all = np.load("data/X.npy")
    y_all = np.load("data/y.npy")
    episodes = np.load("data/episodes.npy") # re-run `python utils.py prepare samples/*` if missing

    # Hold out whole episodes for validation. Frames from the same
    # race are near-duplicates, so a random split leaks them into both
    # train and val and the val loss lies. A held-out race is the
    # honest offline proxy for live driving.
    VAL_EPISODES = ['luigi_raceway4']
    val_mask = np.isin(episodes, VAL_EPISODES)

    print(x_all.shape[0], 'samples -', int((~val_mask).sum()), 'train,',
          int(val_mask.sum()), 'val from', VAL_EPISODES)

    # Training loop variables
    epochs = 100
    freeze_epochs = 10 # train only the head, then fine-tune the whole model
    batch_size = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model().to(device)

    train_set = DrivingDataset(x_all[~val_mask], y_all[~val_mask], augment=True)
    val_set = DrivingDataset(x_all[val_mask], y_all[val_mask])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    head = list(model.backbone.get_classifier().parameters())
    body = [p for p in model.backbone.parameters() if all(p is not h for h in head)]

    best_val_loss = float('inf')
    optimizer = None

    for epoch in range(epochs):
        # transfer learning: head first, then unfreeze and fine-tune
        if epoch == 0:
            for param in body:
                param.requires_grad = False
            optimizer = torch.optim.Adam(head, lr=1e-3)
        elif epoch == freeze_epochs:
            for param in body:
                param.requires_grad = True
            optimizer = torch.optim.Adam([
                {'params': head, 'lr': 1e-4},
                {'params': body, 'lr': 1e-5},
            ])

        model.train()
        if epoch < freeze_epochs:
            model.backbone.eval() # keep batch norm stats frozen too

        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = customized_loss(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_batch)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_loss += customized_loss(model(x_batch), y_batch).item() * len(x_batch)
        val_loss /= len(val_set)

        print(f'epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')

        # save the best model (like the old ModelCheckpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model_weights.pt')
