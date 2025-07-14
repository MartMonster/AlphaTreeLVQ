# Parameters

ds = 24           # downscale factor for probability smoothing
p_threshold = 0.3 # (average) probability threshold

plot = True # plot the results

import torch
import sys
from pathlib import Path
sys.path.insert(0, 'plant-segmentation')
sys.path.insert(0, 'leaves-2025-07-03')
import alvq
from gmlvq import gmlvq
import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_labels(file_path):
    parsed_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip malformed lines
            label = int(parts[0])
            bbox = list(map(float, parts[1:]))  # [x_center, y_center, width, height]
            parsed_data.append((label, bbox))
    return parsed_data

ground = torch.load('plant-segmentation/pix-classifier-alvq-2025-07-09.pt', weights_only=False)

net = torch.load('gmlvq-2025-07-13.pt', weights_only=False)

img = cv2.imread('../Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/test/4e8ef493-3884-4645-82c6-4291a576684d.png')
labels = parse_labels("../Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/test/4e8ef493-3884-4645-82c6-4291a576684d.txt")

def testImage(lvq, img, labels):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_feats = (img[..., 0:3].astype(np.float32) * (1/255)).reshape(-1, 3)

    width = img.shape[1]
    height = img.shape[0]

    with torch.no_grad():
        scalar = ground.probabilities(torch.from_numpy(rgb_feats))
    scalar = scalar[..., 1].reshape(img.shape[0:2])
    # scalar_orig = scalar
    scalar = cv2.resize(scalar.numpy(), (width // ds, height // ds), interpolation=cv2.INTER_AREA)
    scalar = cv2.resize(scalar, (width, height), interpolation=cv2.INTER_LINEAR)
    img[scalar < p_threshold, :] = 0

    errors = []
    
    for count, (label, bbox) in enumerate(labels):
        x_center, y_center, box_width, box_height = bbox
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        box_width = int(box_width * width)
        box_height = int(box_height * height)
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        if x1 < 0 or y1 < 0 or x2 >= width or y2 >= height:
            continue
        for i in range(0, box_width // 16):
            for j in range(0, box_height // 16):
                x = x1 + i * 16
                y = y1 + j * 16
                # extract the 16x16 square from the image
                square = img[y:y+16, x:x+16]
                # if the square does not fit in the image, pad it with zeros
                if square.shape[0] < 16 or square.shape[1] < 16:
                    square = np.pad(square, ((0, 16 - square.shape[0]), (0, 16 - square.shape[1]), (0, 0)), mode='constant')
                square = square.reshape(1, -1)
                # only test if the square is not black
                if square.sum() > 0:
                    label_pred = lvq.forward(torch.from_numpy(square))
                    if label_pred.item() != label:
                        errors.append((label, label_pred.item(), (x, y, x + 16, y + 16)))
    if plot:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
        ax0.imshow(img)
        for label, bbox in labels:
            x_center, y_center, box_width, box_height = bbox
            x_center = int(x_center * img.shape[1])
            y_center = int(y_center * img.shape[0])
            box_width = int(box_width * img.shape[1])
            box_height = int(box_height * img.shape[0])
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            rect = plt.Rectangle((x1, y1), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
            color = 'white' if not label else 'yellow'
            ax0.text(x1, y1 + 100, str(label), color=color, fontsize=12)
            ax0.add_patch(rect)
        ax1.imshow(img)
        for error in errors:
            _, _, (x1, y1, x2, y2) = error
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax1.add_patch(rect)
        plt.show()
    return errors

testImage(net, img, labels)