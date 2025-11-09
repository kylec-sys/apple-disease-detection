# check_overlap.py
import os
from hashlib import md5

def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return md5(f.read()).hexdigest()

train_files = set()
val_files = set()

# 读取训练集所有图片哈希
for root, _, files in os.walk("data/train"):
    for file in files:
        if file.endswith(".JPG"):
            path = os.path.join(root, file)
            train_files.add(get_file_hash(path))

# 读取验证集所有图片哈希
for root, _, files in os.walk("data/val"):
    for file in files:
        if file.endswith(".JPG"):
            path = os.path.join(root, file)
            val_files.add(get_file_hash(path))

# 查找重叠
overlap = train_files & val_files
print(f"Overlapping images: {len(overlap)}")
if overlap:
    print("First few overlapping hashes:")
    for h in list(overlap)[:5]:
        print(h)
