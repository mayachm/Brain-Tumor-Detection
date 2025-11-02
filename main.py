from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# ----------------------------------------
# Model Architecture + Load Weights
# ----------------------------------------
IMAGE_SIZE = 128

# Base model
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                   include_top=False, weights='imagenet')

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze last 3 layers for fine-tuning
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

# Build model
model = Sequential()
model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))  # 4 classes

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# Load trained weights
model.load_weights('models/model1.weights.h5')

# Class labels
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']


# ----------------------------------------
# File Upload Settings
# ----------------------------------------
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------------------------------
# Helper function: Predict tumor type
# ----------------------------------------
def predict_tumor(image_path):
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# ----------------------------------------
# Flask Routes
# ----------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)

            result, confidence = predict_tumor(file_location)
            return render_template(
                'index.html',
                result=result,
                confidence=f"{confidence * 100:.2f}",
                file_path=f'/uploads/{file.filename}'
            )
    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ----------------------------------------
# Run the app
# ----------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
