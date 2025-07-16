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

# img = cv2.imread('../Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/train/2ec83b5a-f1ad-4dae-bb33-773f1eee7923.png')
# labels = parse_labels("../Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/train/2ec83b5a-f1ad-4dae-bb33-773f1eee7923.txt")
lowerH = 30
upperH = 50
lowerS = 70
upperS = 230
def train2d_histogram(lvq, img, labels):
    loss = []
    for count, (label, bbox) in enumerate(labels):
        print(f"Processing {count + 1}/{len(labels)}: {label}")
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
        loss.append(lvq.train(torch.from_numpy(hist.flatten().reshape(1, -1)), torch.tensor([label])))
    return loss

lvq = gmlvq()
features = (upperH - lowerH) * (upperS - lowerS) # 2D histogram features
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
    loss.extend(train2d_histogram(lvq, img, labels))
date = datetime.datetime.now().strftime("%Y-%m-%d")
print(f"Saving model as gmlvq-2d-hist-{date}.pt")
lvq.save(f'gmlvq-2d-hist-{date}.pt')
plt.plot(loss)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Steps')
plt.show()

def make_histograms(img, labels):
    hist_0 = np.zeros((180, 256), dtype=np.float32)
    hist_1 = np.zeros((180, 256), dtype=np.float32)
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
        hls = cv2.cvtColor(box, cv2.COLOR_BGR2HLS)
        # make histogram of H and S values
        hist = cv2.calcHist([hls], [0, 2], None, [180, 256], [0, 180, 0, 256])
        if label == 0:
            hist_0 += hist
        else:
            hist_1 += hist
    return hist_0, hist_1

# h_0 = np.zeros((180, 256), dtype=np.float32)
# h_1 = np.zeros((180, 256), dtype=np.float32)
# directory = Path("../Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/train")
# for count, file_path in enumerate(directory.glob("*.txt")):
#     print(f"Processing {file_path.name} ({count + 1}/{len(list(directory.glob('*.txt')))})")
#     img_path = file_path.with_suffix('.png')
#     if not img_path.exists():
#         continue
#     img = cv2.imread(str(img_path))
#     labels = parse_labels(file_path.with_suffix('.txt'))
#     h_0_local, h_1_local = make_histograms(img, labels)
#     h_0 += h_0_local
#     h_1 += h_1_local
# eps = 1e-10  # small constant to avoid divide-by-zero
# hist_0_norm = h_0 / (h_0.sum() + eps)
# hist_1_norm = h_1 / (h_1.sum() + eps)
# diff = hist_0_norm - hist_1_norm
# # only select area of interest
# diff = diff[lowerH:upperH, lowerS:upperS]
# plt.figure(figsize=(10, 10))
# plt.imshow(diff, origin='lower', aspect='auto',
#            extent=[lowerS, upperS, lowerH, upperH], cmap='bwr', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
# plt.xlabel('Saturation')
# plt.ylabel('Hue')
# plt.title('Normalized Histogram Difference (Label 1 - Label 0)')
# plt.colorbar(label='Probability Difference')
# plt.tight_layout()
# plt.show()

# log_h_0 = np.log1p(h_0)
# log_h_1 = np.log1p(h_1)
# plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.imshow(log_h_0, origin='lower', aspect='auto',
#            extent=[0, 256, 0, 180], cmap='plasma')
# plt.title('Hue vs Saturation (Label = 0)')
# plt.xlabel('Saturation')
# plt.ylabel('Hue')
# plt.colorbar(label='log(1 + Pixel Count)')

# # Plot label 1 histogram
# plt.subplot(1, 2, 2)
# plt.imshow(log_h_1, origin='lower', aspect='auto',
#            extent=[0, 256, 0, 180], cmap='plasma')
# plt.title('Hue vs Saturation (Label = 1)')
# plt.xlabel('Saturation')
# plt.ylabel('Hue')
# plt.colorbar(label='log(1 + Pixel Count)')

# plt.tight_layout()
# plt.show()




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
    healthy_h_hist = [0] * 180
    healthy_s_hist = [0] * 256
    diseased_h_hist = [0] * 180
    diseased_s_hist = [0] * 256
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
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        box = img[y1:y2, x1:x2]
        if box.shape[0] < 1 or box.shape[1] < 1:
            continue
        # convert box to HLS
        hls = cv2.cvtColor(box, cv2.COLOR_RGB2HLS) # H = [0, 180], L = [0, 255], S = [0, 255]
        # make histogram of H values
        h_hist = cv2.calcHist([hls], [0], None, [180], [0, 180])
        # plt.plot(h_hist)
        # make histogram of S values
        s_hist = cv2.calcHist([hls], [2], None, [256], [0, 256])
        if label == 0:
            healthy_h_hist = [h + hl for h, hl in zip(healthy_h_hist, h_hist.flatten())]
            healthy_s_hist = [s + sl for s, sl in zip(healthy_s_hist, s_hist.flatten())]
        else:
            diseased_h_hist = [h + hl for h, hl in zip(diseased_h_hist, h_hist.flatten())]
            diseased_s_hist = [s + sl for s, sl in zip(diseased_s_hist, s_hist.flatten())]
        continue

        # hue = hls[:, :, 0].flatten()
        # saturation = hls[:, :, 2].flatten()

        # Create 2D histogram
        # hist, xedges, yedges = np.histogram2d(hue, saturation, bins=[180, 256], range=[[0, 180], [0, 256]])

        # Logarithmic scaling: Add 1 to avoid log(0)
        # hist_log = np.log1p(hist)

        # Plot 2D histogram
        # plt.imshow(hist_log.T, origin='lower', aspect='auto', extent=[0, 180, 0, 256], cmap='plasma')
        # plt.xlabel('Hue')
        # plt.ylabel('Saturation')
        # plt.title('Hue vs. Saturation (Log Pixel Count)')
        # cbar = plt.colorbar()
        # cbar.set_label('log(1 + Pixel Count)')
        # plt.show()
        

        # plt.plot(s_hist)
        # plt.show()
        continue
        
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
        
    # loss = [item.item() for item in loss]
    # return loss
    return healthy_h_hist, healthy_s_hist, diseased_h_hist, diseased_s_hist

# lvq = gmlvq()
# features = 16*16*3 # 16x16 RGB squares
# prototypes = 2
# initial_protos = torch.randn(prototypes, features)
# prototype_labels = [0, 1]
# mu = torch.randn(features)
# std = torch.rand(features) + 0.1
# lvq.initialize(features, initial_protos, prototype_labels, True, mu, std)
# loss = []
# healthy_s_hist = [0] * 256
# healthy_h_hist = [0] * 180
# diseased_s_hist = [0] * 256
# diseased_h_hist = [0] * 180
# directory = Path("../Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/train")
# for count, file_path in enumerate(directory.glob("*.txt")):
#     print(f"Processing {file_path.name} ({count + 1}/{len(list(directory.glob('*.txt')))})")
#     img_path = file_path.with_suffix('.png')
#     if not img_path.exists():
#         continue
#     img = cv2.imread(str(img_path))
#     labels = parse_labels(file_path.with_suffix('.txt'))
#     healthy_h_local, healthy_s_local, diseased_h_local, diseased_s_local = trainImage(lvq, img, labels)
#     healthy_h_hist = [h + hl for h, hl in zip(healthy_h_hist, healthy_h_local)]
#     healthy_s_hist = [s + sl for s, sl in zip(healthy_s_hist, healthy_s_local)]
#     diseased_h_hist = [h + hl for h, hl in zip(diseased_h_hist, diseased_h_local)]
#     diseased_s_hist = [s + sl for s, sl in zip(diseased_s_hist, diseased_s_local)]
#     # loss.extend(trainImage(lvq, img, labels))
# # date = datetime.datetime.now().strftime("%Y-%m-%d")
# # print(f"Saving model as gmlvq-{date}.pt")
# # lvq.save(f'gmlvq-{date}.pt')

# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
# ax0.plot(healthy_h_hist, label='Healthy H', color='green')
# ax0.plot(healthy_s_hist, label='Healthy S', color='blue')
# ax1.plot(diseased_h_hist, label='Diseased H', color='red')
# ax1.plot(diseased_s_hist, label='Diseased S', color='orange')
# ax0.set_title('Healthy Histogram')
# ax1.set_title('Diseased Histogram')
# plt.show()

