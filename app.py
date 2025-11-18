# app.py
from flask import Flask, request, render_template, jsonify
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

app = Flask(__name__)

# === 加载模型 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/resnet18_apple_v1.pth", map_location=device))
model = model.to(device)
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, 1)

        class_names = ["Black Rot", "Healthy"]
        result = class_names[predicted.item()]
        confidence = float(confidence.item() * 100)

        return jsonify({
            "prediction": result,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
