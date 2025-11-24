# Facial Emotion Recognition using ResNet50V2

A deep learning project for classifying facial emotions into seven categories using transfer learning with ResNet50V2 architecture.

## 📋 Project Overview

This project implements a facial emotion recognition system that can classify images into seven emotional states:
- 😠 Angry
- 🤢 Disgust  
- 😨 Fear
- 😊 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

The model achieves **67.08% accuracy** on the test set using transfer learning with a two-phase training approach.

## 🚀 Features

- **Transfer Learning**: Uses pre-trained ResNet50V2 with ImageNet weights
- **Data Augmentation**: Comprehensive augmentation to improve generalization
- **Two-Phase Training**: Frozen base model training followed by fine-tuning
- **Class Imbalance Handling**: L2 regularization and careful architecture design
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Model Checkpointing**: Automatic saving of best performing model

## 📊 Dataset

**FER2013 Dataset** - 35,887 grayscale facial images (48×48 pixels)
- **Training samples**: 28,709 images
- **Test samples**: 7,178 images
- **7 emotion classes** with inherent class imbalance

### Class Distribution
| Emotion | Train Count | Test Count |
|---------|-------------|------------|
| Angry | 3,995 | 958 |
| Disgust | 436 | 111 |
| Fear | 4,097 | 1,024 |
| Happy | 7,215 | 1,774 |
| Sad | 4,830 | 1,247 |
| Surprise | 3,171 | 831 |
| Neutral | 4,965 | 1,233 |

## 🏗️ Model Architecture

```
Input (224×224×3) 
    ↓
ResNet50V2 Backbone (frozen initially)
    ↓
Global Average Pooling
    ↓
Batch Normalization
    ↓
Dense(512) + ReLU + Dropout(0.5)
    ↓
Batch Normalization
    ↓
Dense(256) + ReLU + Dropout(0.3)
    ↓
Output(7) + Softmax
```

**Model Statistics:**
- Total Parameters: 24,757,255
- Trainable Parameters: 1,187,335
- Non-trainable Parameters: 23,569,920
- Model Size: 94.44 MB

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Required libraries

### Install Dependencies

```bash
pip install tensorflow matplotlib seaborn scikit-learn pandas numpy
```

### Dataset Setup

1. Download the FER2013 dataset
2. Extract the zip file in the project directory
3. Ensure the directory structure is:
```
project/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

## 🎯 Usage

### Training the Model

Run the complete training pipeline:

```python
python facial_emotion_recognition.py
```

### Training Process

The training occurs in two phases:

**Phase 1 - Frozen Base Model (10 epochs):**
- Base ResNet50V2 layers frozen
- Only custom classification head trains
- Learning rate: 0.001

**Phase 2 - Fine-tuning (up to 40 epochs):**
- Top layers of ResNet50V2 unfrozen
- Reduced learning rate: 0.0001
- Early stopping with patience of 15 epochs

### Key Training Parameters

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 220
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 7
```

### Callbacks Used

- **ReduceLROnPlateau**: Reduce learning rate when validation accuracy plateaus
- **EarlyStopping**: Stop training when validation accuracy stops improving
- **ModelCheckpoint**: Save the best model based on validation accuracy

## 📈 Performance

### Test Set Results
- **Overall Accuracy**: 67.08%
- **Overall Loss**: 1.2117

### Class-wise Performance

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 0.57 | 0.62 | 0.59 |
| Disgust | 0.81 | 0.56 | 0.66 |
| Fear | 0.54 | 0.48 | 0.51 |
| Happy | 0.85 | 0.87 | 0.86 |
| Neutral | 0.61 | 0.67 | 0.63 |
| Sad | 0.56 | 0.53 | 0.54 |
| Surprise | 0.80 | 0.78 | 0.79 |

### Best Model
- **Validation Accuracy**: 65.06%
- **Saved as**: `best_emotion_model.h5`

## 📁 Project Structure

```
facial-emotion-recognition/
├── facial_emotion_recognition.py  # Main training script
├── best_emotion_model.h5          # Best model weights
├── FER2013.zip                    # Dataset (not included in repo)
├── train/                         # Training images
├── test/                          # Test images
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🔧 Code Structure

### Main Functions

1. **`extract_dataset(zip_path, extract_to)`** - Extract dataset from zip
2. **`explore_dataset()`** - Analyze dataset statistics
3. **`visualize_samples()`** - Display sample images
4. **`plot_data_distribution()`** - Show class distribution
5. **`create_model()`** - Build ResNet50V2 model
6. **`unfreeze_model(model, base_model)`** - Prepare for fine-tuning
7. **`plot_training_history(history)`** - Visualize training progress
8. **`plot_confusion_matrix()`** - Display classification errors
9. **`visualize_predictions()`** - Show model predictions

### Key Components

**Data Generators:**
- Training: Augmentation + normalization
- Validation/Test: Only normalization

**Model Architecture:**
- Transfer learning with ResNet50V2
- Custom classification head
- Regularization with Dropout and L2

**Training Strategy:**
- Two-phase transfer learning
- Learning rate scheduling
- Early stopping

## 📊 Outputs Generated

The script generates several visualizations:

1. **Sample Images** - Example from each emotion class
2. **Data Distribution** - Bar charts of class frequencies
3. **Training History** - Accuracy and loss curves
4. **Confusion Matrix** - Classification error analysis
5. **Prediction Samples** - Test images with predictions

## 🎮 Making Predictions

### Load Saved Model

```python
from tensorflow.keras.models import load_model

model = load_model('best_emotion_model.h5')
```

### Predict on New Images

```python
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_emotion(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions[0])
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    return emotions[emotion_index], predictions[0][emotion_index]
```

## ⚡ Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled GPU for faster training
2. **Batch Size**: Adjust based on available memory
3. **Data Loading**: Use fast storage (SSD) for better I/O performance
4. **Mixed Precision**: Enable for potential speedup on compatible hardware

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size
   - Use gradient accumulation

2. **Slow Training**
   - Enable GPU acceleration
   - Use data loading optimizations

3. **Poor Performance**
   - Adjust learning rate
   - Modify data augmentation
   - Try different architecture

## 🔮 Future Improvements

- [ ] Address class imbalance with weighted loss
- [ ] Implement attention mechanisms
- [ ] Add real-time webcam inference
- [ ] Extend to video sequence analysis
- [ ] Deploy as web application
- [ ] Add model interpretability visualizations

