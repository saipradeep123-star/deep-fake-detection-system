"""
preprocess_dataset.py
----------------------
Walks a directory of real/fake videos, extracts a fixed-length face
sequence + EAR (blink) sequence per video, and caches the result as a
.npz file so training doesn't need to re-decode video every epoch.

Expected input layout (works for both Celeb-DF and DFDC after you've
organized them - see README.md for how to get/arrange the raw data):

    data/raw_videos/
        real/
            video_0001.mp4
            ...
        fake/
            video_0001.mp4
            ...

Usage:
    python -m src.preprocessing.preprocess_dataset
"""
import os
import glob
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.preprocessing.face_extractor import FaceExtractor


def preprocess_split(label_name, label_value, extractor):
    video_dir = os.path.join(config.RAW_VIDEO_DIR, label_name)
    out_dir = os.path.join(config.PROCESSED_DIR, label_name)
    os.makedirs(out_dir, exist_ok=True)

    video_paths = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        video_paths.extend(glob.glob(os.path.join(video_dir, ext)))

    if not video_paths:
        print(f"[warn] No videos found in {video_dir}. Skipping.")
        return []

    records = []
    for video_path in tqdm(video_paths, desc=f"Processing {label_name}"):
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(out_dir, f"{video_id}.npz")

        if os.path.exists(out_path):
            records.append((out_path, label_value))
            continue

        faces, ears, success = extractor.extract_sequence(video_path)
        if not success:
            print(f"[warn] Could not extract a usable face sequence from {video_path}")
            continue

        np.savez_compressed(out_path, faces=faces, ears=ears, label=label_value)
        records.append((out_path, label_value))

    return records


def main():
    extractor = FaceExtractor()
    all_records = []
    all_records += preprocess_split("real", 0, extractor)
    all_records += preprocess_split("fake", 1, extractor)
    extractor.close()

    if not all_records:
        print("No data was processed. Check data/raw_videos/{real,fake}/ contains videos.")
        return

    # Shuffle and split 70/15/15 into train/val/test, write CSV manifests
    rng = np.random.default_rng(42)
    rng.shuffle(all_records)

    n = len(all_records)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    splits = {
        "train": all_records[:n_train],
        "val": all_records[n_train:n_train + n_val],
        "test": all_records[n_train + n_val:],
    }

    for split_name, records in splits.items():
        csv_path = os.path.join(config.SPLITS_DIR, f"{split_name}.csv")
        with open(csv_path, "w") as f:
            f.write("path,label\n")
            for path, label in records:
                f.write(f"{path},{label}\n")
        print(f"Wrote {len(records)} records to {csv_path}")


if __name__ == "__main__":
    main()
