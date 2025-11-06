import torch
import sys
from pathlib import Path
sys.path.insert(0, 'LVQ')
sys.path.insert(0, 'LVQ/plant-segmentation')
import alvq
import cv2
import numpy as np
import matplotlib.pyplot as plt

ds = 24           # downscale factor for probability smoothing
p_threshold = 0.3 # (average) probability threshold

net = torch.load('LVQ/plant-segmentation/pix-classifier-alvq-2025-07-09.pt', weights_only=False)
directories = [Path("Croptimal/resnet/train/256_class_0"), Path("Croptimal/resnet/train/256_class_1"), Path("Croptimal/resnet/val/256_class_0"), Path("Croptimal/resnet/val/256_class_1")]
for directory in directories:
    for count, file_path in enumerate(directory.glob("*.png")):
        print(f"Processing {file_path.name} ({count + 1}/{len(list(directory.glob('*.png')))})")
        img = cv2.imread(str(file_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_feats = (img[..., 0:3].astype(np.float32) * (1/255)).reshape(-1, 3)
        with torch.no_grad():
            scalar = net.probabilities(torch.from_numpy(rgb_feats))
        width = img.shape[1]
        height = img.shape[0]
        scalar = scalar[..., 1].reshape(img.shape[0:2])
        scalar = cv2.resize(scalar.numpy(), (width // ds, height // ds), interpolation=cv2.INTER_AREA)
        scalar = cv2.resize(scalar, (width, height), interpolation=cv2.INTER_LINEAR)
        img[scalar < p_threshold, :] = 0
        cv2.imwrite(str(file_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))