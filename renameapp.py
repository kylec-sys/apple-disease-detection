import os
import glob

# 遍历所有 JPG 文件
for file in glob.glob("data/train/Apple___healthy/*.JPG"):
    if ' ' in file:
        new_name = file.replace(' ', '_')
        os.rename(file, new_name)
        print(f"Renamed: {file} -> {new_name}")
