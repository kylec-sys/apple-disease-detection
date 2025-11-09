# split_dataset.py
import os
import shutil
from sklearn.model_selection import train_test_split
import glob

def split_data(source_dir, target_train_dir, target_val_dir, val_ratio=0.2):
    classes = ['Apple___healthy', 'Apple___Black_rot']
    
    for cls in classes:
        src_path = os.path.join(source_dir, cls)
        if not os.path.exists(src_path):
            continue
        
        files = glob.glob(os.path.join(src_path, "*.JPG"))
        if len(files) == 0:
            continue
        
        train_files, val_files = train_test_split(files, test_size=val_ratio, random_state=42)
        
        # 创建目标目录
        os.makedirs(os.path.join(target_train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(target_val_dir, cls), exist_ok=True)
        
        # 复制文件
        for f in train_files:
            shutil.copy(f, os.path.join(target_train_dir, cls))
        for f in val_files:
            shutil.copy(f, os.path.join(target_val_dir, cls))

if __name__ == "__main__":
    split_data("data/all", "data/train", "data/val")
