# CITA-AIC (Custom Image Training & Analysis - AI Classifier)

## Project Overview

CITA-AIC is a comprehensive AI Image Classifier project built with TensorFlow that enables users to create custom image recognition models. This initiative presents a specialized solution for training, evaluating, and deploying neural networks capable of recognizing and precisely categorizing images based on user-defined datasets. The system offers complete control over the training pipeline, enabling deployment of highly customized image recognition capabilities tailored to specific domain requirements.

### Key Features

- **Custom Dataset Support**: Train on your own images for any classification task
- **Dual Architecture Options**:
- Transfer Learning (Pre-trained MobileNetV2/ResNet50/EfficientNetB0)
- Custom CNN built from scratch
- **Integrated Webcam Capture Tool**: Collect training data directly from your camera
- **Comprehensive Visualization**: Training metrics, confusion matrices, and prediction analysis
- **Data Augmentation**: Automatic image augmentation for improved model generalization
- **Real-time Prediction**: Fast inference on single images or batches
- **Training Callbacks**: Early stopping, learning rate reduction, and checkpoint saving
- **Interactive Mode**: User-friendly prediction interface

---

## Use Cases

This classifier has been successfully demonstrated with:

- **Facial Expression Recognition** (Happy, Sad, Angry, Neutral, Surprised)
- Object Classification
- Medical Image Analysis
- Product Quality Control
- Wildlife Species Identification
- Document Classification
- Custom Domain-Specific Tasks

---

## Project Structure

```
CITA-AIC/
│
├── config.py                  # Central configuration file
├── model.py                   # Neural network architectures
├── train.py                   # Model training script
├── predict.py                 # Prediction and inference script
├── webcam_capture.py          # Dataset collection tool
├── requirements.txt           # Python dependencies
│
├── data/                      # Dataset directory
│   ├── train/                # Training images (organized by class)
│   ├── test/                 # Testing images (optional)
│   └── raw/                  # Raw unprocessed images
│
├── models/                    # Saved trained models
│   ├── image_classifier.keras
│   └── class_names.txt
│
├── results/                   # Output directory
│   ├── plots/                # Training visualizations
│   └── predictions/          # Prediction results
│
├── utils/                     # Helper modules
│   ├── data_loader.py        # Image loading and preprocessing
│   ├── preprocessor.py       # Dataset utilities
│   └── __init__.py
│
├── logs/                      # TensorBoard logs
└── notebooks/                 # Jupyter notebooks (optional)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Webcam (optional, for data collection)
- 4GB RAM minimum (8GB recommended)
- GPU (optional, for faster training)

### Installation

1. **Clone or download the repository**

   ```bash
   cd CITA-AIC
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   ```

### Basic Usage

#### 1. Collect Your Dataset

**Option A: Use Webcam Capture Tool**

```bash
python webcam_capture.py
```

- Press number keys (1-5) to capture images for different categories
- Aim for 50-100 images per category
- Images are automatically saved and organized

**Option B: Manual Dataset Organization**

```
data/train/
├── category1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── category2/
│   └── ...
└── category3/
    └── ...
```

#### 2. Configure Training Settings

Edit `config.py` to customize:

```python
EPOCHS = 15                          # Training cycles
BATCH_SIZE = 32                      # Images per batch
MODEL_TYPE = 'transfer_learning'     # or 'custom_cnn'
USE_DATA_AUGMENTATION = True         # Enable/disable augmentation
```

#### 3. Train Your Model

```bash
python train.py
```

Training process:

1. ✅ Validates dataset structure
2. ✅ Loads and preprocesses images
3. ✅ Builds neural network architecture
4. ✅ Trains model with progress tracking
5. ✅ Saves best model automatically
6. ✅ Generates performance visualizations

#### 4. Make Predictions

**Single image:**

```bash
python predict.py path/to/image.jpg
```

**Batch prediction:**

```bash
python predict.py path/to/folder/
```

**Interactive mode:**

```bash
python predict.py --interactive
```

---

## 📊 Model Architectures

### Transfer Learning (Recommended)

Uses pre-trained models trained on ImageNet (1.4M images, 1000 categories):

**Available Base Models:**

- **MobileNetV2**: Lightweight, fast, 3.5M parameters (recommended for beginners)
- **ResNet50**: More accurate, 25M parameters
- **EfficientNetB0**: Best balance, 5M parameters

**Architecture:**

```
Pre-trained Base (Frozen/Unfrozen)
    ↓
Global Average Pooling
    ↓
Dense Layer (512 neurons) + Dropout
    ↓
Dense Layer (256 neurons) + Dropout
    ↓
Output Layer (N classes, softmax)
```

**When to use:**

- Small datasets (< 1000 images)
- Limited computational resources
- Need faster training
- Want higher accuracy with less data

### Custom CNN (From Scratch)

Build your own convolutional neural network:

**Architecture:**

```
Input (224x224x3)
    ↓
Conv2D (32 filters) → MaxPool → Dropout
    ↓
Conv2D (64 filters) → MaxPool → Dropout
    ↓
Conv2D (128 filters) → MaxPool → Dropout
    ↓
Conv2D (128 filters) → MaxPool → Dropout
    ↓
Flatten
    ↓
Dense (256) → Dropout
    ↓
Dense (128) → Dropout
    ↓
Output (N classes, softmax)
```

**When to use:**

- Large datasets (> 2000 images)
- Learning purposes
- Need full control over architecture
- Domain-specific requirements

---

## ⚙️ Configuration Guide

### Key Parameters in `config.py`

| Parameter                 | Default    | Description               | Recommendation                     |
| ------------------------- | ---------- | ------------------------- | ---------------------------------- |
| `IMG_SIZE`                | (224, 224) | Input image dimensions    | Keep default for transfer learning |
| `BATCH_SIZE`              | 32         | Images per training batch | Reduce if out of memory            |
| `EPOCHS`                  | 15         | Training iterations       | Increase for better accuracy       |
| `LEARNING_RATE`           | 0.001      | Optimizer learning rate   | Lower for fine-tuning              |
| `VALIDATION_SPLIT`        | 0.2        | Data for validation       | 0.2 = 20% validation               |
| `DROPOUT_RATE`            | 0.4        | Regularization strength   | Increase if overfitting            |
| `USE_DATA_AUGMENTATION`   | True       | Enable augmentation       | Recommended for small datasets     |
| `EARLY_STOPPING_PATIENCE` | 5          | Epochs before stopping    | Prevents overfitting               |

### Data Augmentation Settings

```python
AUGMENTATION_CONFIG = {
    'rotation_range': 20,        # Rotate ±20 degrees
    'width_shift_range': 0.2,    # Shift horizontally 20%
    'height_shift_range': 0.2,   # Shift vertically 20%
    'horizontal_flip': True,     # Random horizontal flip
    'zoom_range': 0.2,           # Zoom in/out 20%
    'shear_range': 0.1,          # Shear transformation
    'fill_mode': 'nearest'       # Fill strategy
}
```

---

## 📈 Training Process

### Monitoring Training

Watch for these metrics during training:

**Good Training:**

```
Epoch 10/15
accuracy: 0.92, val_accuracy: 0.89  ✅ Small gap
loss: 0.25, val_loss: 0.30
```

**Overfitting (Bad):**

```
Epoch 10/15
accuracy: 0.98, val_accuracy: 0.65  ❌ Large gap
loss: 0.10, val_loss: 0.85
```

### Training Callbacks

The system automatically:

- **Saves best model** based on validation accuracy
- **Stops early** if no improvement for N epochs
- **Reduces learning rate** when validation plateaus
- **Logs metrics** for TensorBoard visualization

### Expected Training Times

| Dataset Size | Model Type        | Hardware | Approximate Time |
| ------------ | ----------------- | -------- | ---------------- |
| 500 images   | Transfer Learning | CPU      | 10-15 minutes    |
| 500 images   | Custom CNN        | CPU      | 30-45 minutes    |
| 2000 images  | Transfer Learning | CPU      | 30-60 minutes    |
| 2000 images  | Transfer Learning | GPU      | 5-10 minutes     |
| 2000 images  | Custom CNN        | GPU      | 15-30 minutes    |

---

## 🎯 Making Predictions

### Command-Line Usage

```bash
# Single image prediction
python predict.py image.jpg

# Folder of images
python predict.py images_folder/

# Multiple specific images
python predict.py img1.jpg img2.jpg img3.jpg

# Interactive mode
python predict.py --interactive
```

### Programmatic Usage

```python
from predict import load_trained_model, load_class_names, predict_single_image

# Load once
model = load_trained_model()
class_names = load_class_names()

# Predict multiple times
result = predict_single_image(model, 'image.jpg', class_names, show_plot=False)

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}%")

# Access all probabilities
for class_name, prob in result['all_probabilities'].items():
    print(f"{class_name}: {prob*100:.2f}%")
```

### Prediction Output

```
✓ Prediction complete!
  Predicted class: happy
  Confidence: 95.43%

  All class probabilities:
    happy: 95.43%
    neutral: 2.31%
    sad: 1.42%
    surprised: 0.58%
    angry: 0.26%
```

---

## 🛠️ Advanced Features

### Fine-Tuning

Improve accuracy after initial training:

```python
from model import unfreeze_base_model
from tensorflow import keras

# Load trained model
model = keras.models.load_model('models/image_classifier.keras')

# Unfreeze last 20 layers
model = unfreeze_base_model(model, layers_to_unfreeze=20)

# Continue training with lower learning rate
# (automatically set by unfreeze_base_model)
```

### Dataset Management

**Split raw data into train/test:**

```python
from utils.preprocessor import split_dataset

split_dataset(
    source_dir='data/raw',
    train_dir='data/train',
    test_dir='data/test',
    test_split=0.2
)
```

**Clean dataset:**

```python
from utils.preprocessor import clean_dataset

stats = clean_dataset(
    'data/train',
    min_size=(50, 50),
    remove_invalid=True
)
```

**Analyze dataset:**

```python
from utils.preprocessor import analyze_dataset

analyze_dataset('data/train')
```

### TensorBoard Visualization

View detailed training metrics:

```bash
tensorboard --logdir=logs/fit
```

Then open `http://localhost:6006` in your browser.

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Error

**Problem:** GPU/RAM runs out of memory

**Solutions:**

```python
# In config.py
BATCH_SIZE = 16  # Reduce from 32
IMG_SIZE = (128, 128)  # Reduce from (224, 224)
```

#### 2. Model Not Learning

**Problem:** Accuracy stays low (< 40%)

**Solutions:**

- Check dataset organization (correct folder structure)
- Increase training epochs: `EPOCHS = 25`
- Use transfer learning: `MODEL_TYPE = 'transfer_learning'`
- Ensure sufficient data (50+ images per class)

#### 3. Overfitting

**Problem:** High training accuracy, low validation accuracy

**Solutions:**

```python
USE_DATA_AUGMENTATION = True
DROPOUT_RATE = 0.5  # Increase
EPOCHS = 10  # Reduce
```

#### 4. Low Confidence Predictions

**Problem:** Model predicts with < 60% confidence

**Solutions:**

- Collect more training data
- Enable data augmentation
- Train for more epochs
- Clean dataset (remove ambiguous images)
- Use transfer learning

#### 5. Webcam Not Working

**Problem:** `webcam_capture.py` fails to open camera

**Solutions:**

- Close other applications using webcam
- Check camera permissions
- Try different camera: Press 'S' to switch
- Install: `pip install opencv-python --upgrade`

---

## 📊 Performance Optimization

### For Small Datasets (< 500 images)

```python
MODEL_TYPE = 'transfer_learning'
FREEZE_BASE_MODEL = True
USE_DATA_AUGMENTATION = True
EPOCHS = 20
DROPOUT_RATE = 0.5
```

### For Large Datasets (> 2000 images)

```python
MODEL_TYPE = 'transfer_learning'
FREEZE_BASE_MODEL = False  # Train all layers
USE_DATA_AUGMENTATION = False
EPOCHS = 10
BATCH_SIZE = 64
```

### For GPU Acceleration

```python
BATCH_SIZE = 64  # Increase for GPU
EPOCHS = 10  # Reduce (GPU trains faster)
```

---

## 📝 Best Practices

### Data Collection

✅ **Do:**

- Collect 50-200 images per category
- Vary lighting, angles, and backgrounds
- Use clear, focused images
- Balance classes (similar number of images per category)
- Include edge cases and variations

❌ **Don't:**

- Use blurry or low-quality images
- Have huge class imbalances
- Include mislabeled images
- Use exact duplicates

### Training

✅ **Do:**

- Start with transfer learning
- Monitor validation metrics
- Use early stopping
- Save multiple checkpoints
- Validate on unseen data

❌ **Don't:**

- Train on test data
- Ignore overfitting signs
- Use tiny learning rates (< 1e-6)
- Stop training too early

### Deployment

✅ **Do:**

- Test on diverse real-world images
- Set appropriate confidence thresholds
- Handle edge cases gracefully
- Monitor prediction distribution
- Version your models

❌ **Don't:**

- Deploy without testing
- Assume 100% accuracy
- Ignore low-confidence predictions
- Use overfitted models

---

## 🎓 Learning Resources

### Understanding the Code

Each Python file contains extensive comments explaining:

- What the code does
- Why we do it
- How it works
- When to use different options

**Recommended Reading Order:**

1. `config.py` - Understand all settings
2. `model.py` - Learn neural network architectures
3. `utils/data_loader.py` - Image preprocessing
4. `train.py` - Training pipeline
5. `predict.py` - Inference and prediction

### External Resources

**Neural Networks Basics:**

- [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [CNN Explainer (Interactive)](https://poloclub.github.io/cnn-explainer/)

**TensorFlow Tutorials:**

- [Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

**Computer Vision:**

- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Stanford CS231n](http://cs231n.stanford.edu/)

---

## 📦 Dependencies

```
tensorflow==2.15.0
numpy==1.24.3
matplotlib==3.7.1
pillow==10.0.0
scikit-learn==1.3.0
pandas==2.0.3
opencv-python==4.8.0  # For webcam capture
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional pre-trained model support
- Mobile deployment (TensorFlow Lite)
- Web interface (Flask/FastAPI)
- Real-time video classification
- Model explainability (Grad-CAM)
- Multi-label classification support
- AutoML hyperparameter tuning

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **TensorFlow Team** - For the amazing deep learning framework
- **ImageNet** - For pre-trained model weights
- **OpenCV** - For webcam capture functionality
- **Keras Applications** - For pre-trained architectures

---

## 📧 Contact & Support

For questions, issues, or suggestions:

- Open an issue on GitHub
- Check the [Troubleshooting](#troubleshooting) section
- Review code comments for detailed explanations

---

## 🎯 Project Status

**Current Version:** 1.0.0  
**Status:** ✅ Active Development  
**Last Updated:** December 2025

### Completed Features

- ✅ Transfer learning support (MobileNetV2, ResNet50, EfficientNetB0)
- ✅ Custom CNN architecture
- ✅ Webcam dataset collection tool
- ✅ Data augmentation
- ✅ Training visualization
- ✅ Batch prediction
- ✅ Interactive prediction mode
- ✅ Comprehensive documentation

### Roadmap

- 🔄 Real-time video classification
- 🔄 Web interface deployment
- 🔄 Mobile app support (TensorFlow Lite)
- 🔄 Model explainability visualization
- 🔄 AutoML integration
- 🔄 Multi-label classification

---

## 📊 Example Results

### Facial Expression Recognition Demo

**Dataset:** 250 images (50 per emotion)  
**Model:** Transfer Learning (MobileNetV2)  
**Training Time:** 12 minutes (CPU)  
**Final Accuracy:** 87% (training), 44% (validation)

**Sample Predictions:**

```
Image: happy_expression.jpg
Predicted: happy (47.89% confidence)
✅ Correct

Image: sad_expression.jpg
Predicted: sad (68.23% confidence)
✅ Correct

Image: angry_expression.jpg
Predicted: angry (82.14% confidence)
✅ Correct
```

---

## 🔥 Quick Tips

1. **Start Small:** Begin with 2-3 classes and 50 images each
2. **Use Transfer Learning:** Much faster and more accurate for small datasets
3. **Enable Augmentation:** Always use for datasets < 1000 images
4. **Monitor Overfitting:** Watch the gap between train and validation accuracy
5. **Test Often:** Predict on new images frequently during development
6. **Good Lighting:** Makes huge difference in webcam capture quality
7. **Vary Your Data:** Don't capture all images in one session
8. **Save Checkpoints:** Training can be interrupted; callbacks save progress

---

**Built with TensorFlow and Python**
