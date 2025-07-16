# Parameters

ds = 24           # downscale factor for probability smoothing
p_threshold = 0.3 # (average) probability threshold

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
import datetime

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

net = torch.load('plant-segmentation/pix-classifier-alvq-2025-07-09.pt', weights_only=False)
lowerH = 30
upperH = 50
lowerS = 70
upperS = 230
def test_2d_histogram(lvq, img, labels):
    errors = []
    for count, (label, bbox) in enumerate(labels):
        x_center, y_center, box_width, box_height = bbox
        x_center = int(x_center * img.shape[1])
        y_center = int(y_center * img.shape[0])
        box_width = int(box_width * img.shape[1])
        box_height = int(box_height * img.shape[0])
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        box = img[y1:y2, x1:x2]
        if box.shape[0] < 1 or box.shape[1] < 1:
            continue
        # convert box to HLS
        hls = cv2.cvtColor(box, cv2.COLOR_RGB2HLS)
        # make histogram of H and S values
        hist = cv2.calcHist([hls], [0, 2], None, [180, 256], [0, 180, 0, 256])
        # crop the histogram to the area of interest
        hist = hist[lowerH:upperH, lowerS:upperS]
        label_pred = lvq.forward(torch.from_numpy(hist.flatten().reshape(1, -1)))
        if label_pred.item() != label:
            errors.append((label, label_pred.item(), (x1, y1, x2, y2)))
            print(f"Error: Expected {label}, got {label_pred.item()} at ({count + 1}/{len(labels)})")
    return errors

plot = False

lvq = torch.load('gmlvq-2d-hist-2025-07-15.pt', weights_only=False)
errors = []
ones = 0
zeros = 0
directory = Path("../Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/train")
for count, file_path in enumerate(directory.glob("*.txt")):
    print(f"Processing {file_path.name} ({count + 1}/{len(list(directory.glob('*.txt')))})")
    img_path = file_path.with_suffix('.png')
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))

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

    labels = parse_labels(file_path.with_suffix('.txt'))
    errors.extend(test_2d_histogram(lvq, img, labels))
    # extract number of 1s and 0s from labels
    ones += sum(1 for label, _ in labels if label == 1)
    zeros += sum(1 for label, _ in labels if label == 0)
    
    if plot:
        _, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
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
        for label, _, (x1, y1, x2, y2) in errors[-1]:
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none'))
        plt.title(f"Errors: {len(errors[-1])}")
        plt.show()
# false positive rates
fpr = sum(1 for label, _, _ in errors if label == 0) / zeros if zeros > 0 else 0
fnr = sum(1 for label, _, _ in errors if label == 1) / ones if ones > 0 else 0
error_rate = len(errors) / (ones + zeros) if (ones + zeros) > 0 else 0
print(f"False Positive Rate: {fpr}\n\nFalse Negative Rate: {fnr}\n\nError Rate: {error_rate}")