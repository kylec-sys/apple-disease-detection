# predict.py
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys
import os

# 检查参数
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: File {image_path} not found!")
    sys.exit(1)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型结构（必须和训练时一致）
model = models.resnet18(weights=None)  # 不加载预训练权重
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 类：Black_rot / Healthy

# 加载训练好的权重
model.load_state_dict(torch.load("models/resnet18_apple_v1.pth", map_location=device))
model = model.to(device)
model.eval()

# 图像预处理（必须和训练时完全一致！）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载并预测
img = Image.open(image_path).convert("RGB")  # 确保是 RGB
input_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    prob = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(prob, 1)

# 类别标签（顺序必须和 ImageFolder 一致！）
class_names = ["Apple___Black_rot", "Apple___healthy"]
result = "Black Rot" if predicted.item() == 0 else "Healthy"
confidence = confidence.item() * 100

print(f"✅ Prediction: {result}")
print(f"📊 Confidence: {confidence:.2f}%")
