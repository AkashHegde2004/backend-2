# CNN Face Recognition Training

This document explains how to use the CNN-based face recognition training functionality that has been added to improve student classification accuracy.

## Overview

The system now includes a CNN (Convolutional Neural Network) training module that can be used to train a more accurate face recognition model. The CNN model is trained on face embeddings extracted using DeepFace and saved as a .h5 file for later use.

## How It Works

1. The system extracts face embeddings from student images using DeepFace with the Facenet512 model
2. These embeddings are used as features to train a CNN model
3. The trained CNN model is saved as `cnn_face_recognition_model.h5`
4. The label encoder is saved as `cnn_face_recognition_model_label_encoder.pkl`
5. During recognition, the system first tries the CNN model, falling back to the traditional method if needed

## Training the CNN Model

### Method 1: Using the API Endpoint

You can train the CNN model by calling the `/api/train-cnn-model` endpoint:

```bash
curl -X POST http://localhost:8000/api/train-cnn-model
```

### Method 2: Using the Training Script

You can also run the training script directly:

```bash
cd backend
python cnn_training.py
```

### Method 3: Comprehensive Training

For a complete training that includes both traditional embeddings and CNN model:

```bash
cd backend
python comprehensive_training.py
```

## Model Architecture

The CNN model uses the following architecture:
- Input layer matching the dimension of Facenet512 embeddings (512 features)
- Dense layer with 512 neurons and ReLU activation
- Dropout layer (50%)
- Dense layer with 256 neurons and ReLU activation
- Dropout layer (50%)
- Dense layer with 128 neurons and ReLU activation
- Dropout layer (30%)
- Output layer with softmax activation for classification

## Improving Accuracy

To improve the accuracy of student classification:

1. Ensure each student has multiple high-quality images (at least 5-10 images per student)
2. Images should show the face from different angles and under different lighting conditions
3. Faces should be clearly visible and well-aligned
4. Avoid images with accessories that might obscure the face (sunglasses, masks, etc.)
5. Retrain the model after adding new students or images

## Troubleshooting

If students are not being classified correctly:

1. Check that the CNN model files exist:
   - `cnn_face_recognition_model.h5`
   - `cnn_face_recognition_model_label_encoder.pkl`

2. Verify that students have sufficient training images in the `uploads/students` directory

3. Retrain the model using the comprehensive training script:
   ```bash
   python comprehensive_training.py
   ```

4. Adjust the confidence threshold in the services.py file if needed