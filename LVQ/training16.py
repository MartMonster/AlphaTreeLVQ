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

img = cv2.imread('../Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/train/2ec83b5a-f1ad-4dae-bb33-773f1eee7923.png')
labels = parse_labels("../Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/train/2ec83b5a-f1ad-4dae-bb33-773f1eee7923.txt")

def trainImage(lvq, img, labels):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rgb_feats = (img[..., 0:3].astype(np.float32) * (1/255)).reshape(-1, 3)

    with torch.no_grad():
        scalar = net.probabilities(torch.from_numpy(rgb_feats))

    width = img.shape[1]
    height = img.shape[0]

    scalar = scalar[..., 1].reshape(img.shape[0:2])
    # scalar_orig = scalar
    scalar = cv2.resize(scalar.numpy(), (width // ds, height // ds), interpolation=cv2.INTER_AREA)
    scalar = cv2.resize(scalar, (width, height), interpolation=cv2.INTER_LINEAR)
    img[scalar < p_threshold, :] = 0
    loss = []
    # fix, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(img)
    for count, (label, bbox) in enumerate(labels):
        x_center, y_center, box_width, box_height = bbox
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        box_width = int(box_width * width)
        box_height = int(box_height * height)
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        # x2 = int(x_center + box_width / 2)
        # y2 = int(y_center + box_height / 2)
        
        # rect = plt.Rectangle((x1, y1), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # color = 'white' if not label else 'yellow'
        # ax.text(x1, y1 + 100, str(label), color=color, fontsize=12)
        # add 16x16 squares in the rectangle until the rectangle is filled
        for i in range(0, box_width // 16):
            for j in range(0, box_height // 16):
                x = x1 + i * 16
                y = y1 + j * 16
                # rect = plt.Rectangle((x, y), 16, 16, linewidth=0.1, edgecolor='blue', facecolor='none')
                # ax.add_patch(rect)
                # extract the 16x16 square from the image
                square = img[y:y+16, x:x+16]
                # if the square does not fit in the image, pad it with zeros
                if square.shape[0] < 16 or square.shape[1] < 16:
                    square = np.pad(square, ((0, 16 - square.shape[0]), (0, 16 - square.shape[1]), (0, 0)), mode='constant')
                square = square.reshape(1, -1)
                # only train if the square is not black
                if square.sum() > 0:
                    loss.append(lvq.train(torch.from_numpy(square), torch.tensor([label])))
        print(f"({count+1}/{len(labels)}) loss: {loss[-1].item()}, label: {label}")
        
    loss = [item.item() for item in loss]
    return loss

lvq = gmlvq()
features = 16*16*3 # 16x16 RGB squares
prototypes = 2
initial_protos = torch.randn(prototypes, features)
prototype_labels = [0, 1]
mu = torch.randn(features)
std = torch.rand(features) + 0.1
lvq.initialize(features, initial_protos, prototype_labels, True, mu, std)
loss = []
directory = Path("../Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/train")
for count, file_path in enumerate(directory.glob("*.txt")):
    print(f"Processing {file_path.name} ({count + 1}/{len(list(directory.glob('*.txt')))})")
    img_path = file_path.with_suffix('.png')
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    labels = parse_labels(file_path.with_suffix('.txt'))
    loss.extend(trainImage(lvq, img, labels))
lvq.save('gmlvq-2025-07-13.pt')

plt.figure(figsize=(10, 10))
plt.plot(loss)
plt.show()

