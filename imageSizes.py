import os
from PIL import Image
import pandas as pd
from statistics import mode, StatisticsError

def get_image_dimensions(folder_path):
    dimensions = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    dimensions.append(img.size)  # (width, height)
            except Exception as e:
                print(f"Could not process {filename}: {e}")
    return dimensions

def summarize_dimensions(dimensions):
    if not dimensions:
        return None
    
    df = pd.DataFrame(dimensions, columns=['width', 'height'])
    summary = {
        'width_min': df['width'].min(),
        'width_max': df['width'].max(),
        'width_mean': df['width'].mean(),
        'width_median': df['width'].median(),
        'height_min': df['height'].min(),
        'height_max': df['height'].max(),
        'height_mean': df['height'].mean(),
        'height_median': df['height'].median(),
    }

    # Handle mode (can fail if all values are unique)
    try:
        summary['width_mode'] = mode(df['width'])
    except StatisticsError:
        summary['width_mode'] = None
    try:
        summary['height_mode'] = mode(df['height'])
    except StatisticsError:
        summary['height_mode'] = None

    return summary

# Paths to your folders
folder1 = "Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/train/class_0/"
folder2 = "Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/train/class_1/"

# Get dimensions
dims1 = get_image_dimensions(folder1)
dims2 = get_image_dimensions(folder2)

# Summarize
summary1 = summarize_dimensions(dims1)
summary2 = summarize_dimensions(dims2)

# Display as DataFrame
summary_df = pd.DataFrame([summary1, summary2], index=['class_0', 'class_1'])
print(summary_df)
