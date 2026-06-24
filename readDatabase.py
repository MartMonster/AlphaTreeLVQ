import sqlite3
import os

DATA_DIR = "Croptimal/2024_5_13_CleansingDataset/Run1_light_normal_otherobjects/"
dirs = [os.path.join(DATA_DIR, "train"), os.path.join(DATA_DIR, "test"), os.path.join(DATA_DIR, "val")]

for d in dirs:
    imgs = [f"select Variety from Photos join Measurements on Photos.MeasurementId = Measurements.MeasurementId where Filename is '{f}'" for f in os.listdir(d) if f.lower().endswith('.png')]
    dbfile = "Croptimal/2024_5_13_CleansingDataset/Database.db"
    con = sqlite3.connect(dbfile)
    cur = con.cursor()

    variety_counts = {}
    for query in imgs:
        cur.execute(query)
        result = cur.fetchone()
        if result:
            variety = result[0]
            if variety in variety_counts:
                variety_counts[variety] += 1
            else:
                variety_counts[variety] = 1

    print(f"Variety counts for {d}:")
    for variety, count in variety_counts.items():
        print(f"  {variety}: {count} ({count / len(imgs) * 100:.2f}%)")

    con.close()