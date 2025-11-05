                                                          Brain Tumor Detection Using Deep Learning.
The model is trained to classify four categories: Glioma, Meningioma, Pituitary tumor, and No Tumor.
The goal is to support medical research and assist in early tumor identification through automated imaging analysis.
This system is not a medical diagnostic tool. It is a research and learning project demonstrating the application of CNNs (Convolutional Neural Networks) in medical imaging.

Preprocessing and data normalization:
Trained model saved and loaded for inference
Flask web app for uploading MRI scans and receiving predictions with confidence scores

Dataset:
The model is trained using publicly available brain MRI tumor datasets.
Images cover specific tumor types and non-tumorous brains to allow accurate multi-class learning.

Tech Stack:
Python
TensorFlow / Keras
NumPy, Matplotlib, Pandas
Flask (web interface)

Future Enhancements:
Improve model accuracy with more diverse datasets
Add Grad-CAM heatmaps to highlight tumor regions
Integrate a medical image viewer
Deploy on cloud for real-time inference
