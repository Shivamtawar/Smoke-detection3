import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, request, render_template, jsonify
import os
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 classes: Fire, Smoke, Neutral
model.load_state_dict(torch.load('weights/model_best.pth', map_location=torch.device('cpu')))
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define transforms
transform = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(img_path):
    start_time = time.time()
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        return None, f"Error loading image: {e}", 0
    
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        class_names = ['Fire', 'Smoke', 'Neutral']
        predicted_class = class_names[predicted.item()]
    
    end_time = time.time()
    total_time = end_time - start_time
    return predicted_class, confidence.item(), total_time

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file selected')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No file selected')
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            predicted_class, confidence, total_time = predict_image(filepath)
            if predicted_class is None:
                return render_template('upload.html', error=confidence)
            return render_template('result.html', prediction=predicted_class, confidence=f"Confidence score: {confidence:.4f}", time_str=f"Inference executed in {total_time:.2f}s", image_path=filepath)
        return render_template('upload.html', error='Invalid file format. Use PNG, JPG, or JPEG.')
    return render_template('upload.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_class, confidence, total_time = predict_image(filepath)
        if predicted_class is None:
            return jsonify({'error': confidence}), 400
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.4f}",
            'inference_time': f"{total_time:.2f}"
        })
    return jsonify({'error': 'Invalid file format. Use PNG, JPG, or JPEG.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)