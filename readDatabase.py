import sqlite3
import os

DATA_DIR2 = "Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/"
folders2 = ["train", "test", "val"]
DATA_DIR = "Croptimal/2024_5_13_CleansingDataset/"
folders = ["Bladrol", "Dark", "Light", "Normal", "OtherObjects"]
dirs = [os.path.join(DATA_DIR, folder) for folder in folders]
dirs2 = [os.path.join(DATA_DIR2, folder) for folder in folders2]
dirs.extend(dirs2)

dbfile = "Croptimal/2024_5_13_CleansingDataset/Database.db"
con = sqlite3.connect(dbfile)
cur = con.cursor()
for d in dirs:
    imgs = [f"select Variety, Photos.MeasurementId from Photos join Measurements on Photos.MeasurementId = Measurements.MeasurementId where Filename is '{f}'" for f in os.listdir(d) if f.lower().endswith('.png')]

    variety_counts = {}
    measurement_ids = {}
    for query in imgs:
        cur.execute(query)
        result = cur.fetchone()
        if result:
            variety = result[0]
            measurement_id = result[1]
            if variety in variety_counts:
                variety_counts[variety] += 1
            else:
                variety_counts[variety] = 1
            if measurement_id in measurement_ids:
                measurement_ids[measurement_id] += 1
            else:
                measurement_ids[measurement_id] = 1

    print(f"Variety counts for {d}:")
    for variety, count in variety_counts.items():
        print(f"  {variety}: {count} ({count / len(imgs) * 100:.2f}%)")

    print(f"Measurement ID counts for {d}:")
    for measurement_id, count in measurement_ids.items():
        print(f"  {measurement_id}: {count} ({count / len(imgs) * 100:.2f}%)")

    print()
con.close()