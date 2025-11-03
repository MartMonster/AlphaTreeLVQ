from pathlib import Path
import cv2

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

directory = Path("Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/test")
for count, file_path in enumerate(directory.glob("*.txt")):
    print(f"Processing {file_path.name} ({count + 1}/{len(list(directory.glob('*.txt')))})")
    img_path = file_path.with_suffix('.png')
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    labels = parse_labels(file_path)
    for index, (label, bbox) in enumerate(labels):
        if label != 0 and label != 1:
            continue
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
        save_dir = directory / f"class_{label}"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{file_path.stem}_{index}.png"
        if box.shape[0] < 1 or box.shape[1] < 1:
            continue
        cv2.imwrite(str(save_path), box)