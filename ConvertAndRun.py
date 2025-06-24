import sys
import cv2
import numpy as np

if (len(sys.argv) != 2):
    print("program use: python3 [filename] [image input name]")
    sys.exit(1)
imageName = sys.argv[1]
print("input image: "+imageName)
img = cv2.imread(imageName, cv2.IMREAD_UNCHANGED)
if img is None:
    raise ValueError("Could not read or find image")
if img.dtype == np.uint16:
    # Scale 8-bit values to 16-bit range (0–255 to 0–65535)
    img_8bit = (img / 257).astype(np.uint8)
else:
    img_8bit = img

def apply_clahe(image_8bit):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # If grayscale
    if len(image_8bit.shape) == 2:
        equalized = clahe.apply(image_8bit)
        return (equalized.astype(np.uint16) * 256)

    # If color image (apply CLAHE to each channel)
    channels = cv2.split(image_8bit)
    equalized_channels = [clahe.apply(ch) for ch in channels]
    merged = cv2.merge(equalized_channels)

    # Convert back to 16-bit
    return (merged.astype(np.uint16) * 256)

img_normalized_16bit = apply_clahe(img_8bit)
cv2.imwrite("output/normalized_16bpc.png", img_normalized_16bit)

#TODO: write config file and execute alphatree algorithm from this program