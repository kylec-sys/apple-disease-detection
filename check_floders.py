# check_folders.py
import os

train_dir = "data/train"
val_dir = "data/val"

print("Train folders:")
for folder in os.listdir(train_dir):
    print(f"  {folder}")

print("\nVal folders:")
for folder in os.listdir(val_dir):
    print(f"  {folder}")
