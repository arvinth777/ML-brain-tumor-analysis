# 🧠 Brain Tumor Detection using Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered web application for detecting brain tumors from MRI scans using deep learning and computer vision techniques.

## 📋 Overview

This project implements a brain tumor detection system that analyzes MRI brain scans and classifies them as either containing a tumor or being tumor-free. The system uses transfer learning with ResNet50 for feature extraction and a Linear SVM classifier for prediction.

### Key Features

- 🖼️ **Easy Image Upload**: Drag and drop MRI scan images
- 🤖 **AI-Powered Analysis**: Uses ResNet50 + Linear SVM for accurate predictions
- 📊 **Confidence Scores**: Displays prediction confidence percentage
- 🎨 **Modern UI**: Clean, intuitive interface built with Streamlit
- ⚡ **Fast Processing**: Real-time predictions in seconds
- 📱 **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Model Architecture

### Feature Extraction
- **Model**: ResNet50 (pre-trained on ImageNet)
- **Input Size**: 224x224x3
- **Output**: 2048-dimensional feature vector

### Classification
- **Classifier**: Linear Support Vector Machine (SVM)
- **Training Data**: Br35H-Mask-RCNN dataset
- **Classes**: Binary (Tumor / No Tumor)

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.67% |
| **Precision** | 100.00% |
| **Recall** | 97.33% |
| **F1 Score** | 98.65% |
| **ROC AUC** | 99.77% |

## 📊 Dataset

The model was trained on the **Br35H-Mask-RCNN** dataset from Kaggle:

- **Source**: [Brain Tumor Detection Dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
- **Total Images**: 3,000 MRI scans
- **Classes**: 
  - Tumor (Yes): 1,500 images
  - No Tumor (No): 1,500 images
- **Format**: JPG images
- **Split**: 70% training, 15% validation, 15% testing

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/arvinth777/ML-brain-tumor-analysis.git
   cd ML-brain-tumor-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running Locally

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

3. **Use the application**
   - Upload an MRI brain scan image (JPG, JPEG, or PNG)
   - Click "Analyze Image"
   - View the prediction results and confidence score

### Using the Web App

1. Visit the [live demo](https://your-app-url.streamlit.app)
2. Upload your MRI scan
3. Get instant predictions!

## 📁 Project Structure

```
ML-brain-tumor-analysis/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore file
├── .streamlit/
│   └── config.toml                # Streamlit configuration
└── FML_Assignment_Brain_Tumor_2024 (1).ipynb  # Original notebook
```

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[TensorFlow](https://www.tensorflow.org/)** - Deep learning framework
- **[ResNet50](https://keras.io/api/applications/resnet/)** - Pre-trained CNN model
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning library
- **[OpenCV](https://opencv.org/)** - Image processing
- **[Pillow](https://python-pillow.org/)** - Image handling
- **[NumPy](https://numpy.org/)** - Numerical computing

## 🔬 How It Works

1. **Image Upload**: User uploads an MRI brain scan
2. **Preprocessing**: Image is resized to 224x224 and normalized
3. **Feature Extraction**: ResNet50 extracts 2048 features from the image
4. **Classification**: Linear SVM classifies the features
5. **Results**: Prediction and confidence score are displayed

## 📈 Model Training

The model was trained using the following approach:

1. **Data Loading**: Load MRI images from the Br35H-Mask-RCNN dataset
2. **Feature Extraction**: Use ResNet50 to extract features from all images
3. **Model Training**: Train a Linear SVM on the extracted features
4. **Evaluation**: Test on held-out validation set
5. **Optimization**: Fine-tune hyperparameters for best performance

For detailed training code, see the Jupyter notebook: `FML_Assignment_Brain_Tumor_2024 (1).ipynb`

## 🔮 Future Improvements

- [ ] Multi-class classification (tumor types)
- [ ] Tumor segmentation and localization
- [ ] Model interpretability (Grad-CAM visualizations)
- [ ] Support for DICOM medical image format
- [ ] Batch processing for multiple images
- [ ] Model ensemble for improved accuracy
- [ ] Integration with medical imaging systems

## ⚠️ Disclaimer

**This application is for educational and demonstration purposes only.** It should **NOT** be used for actual medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Ahmed Hamada](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection) on Kaggle
- ResNet50: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Streamlit: For the amazing web framework

## 👨‍💻 Author

**Arvinth Cinmayan G K**

- GitHub: [@arvinth777](https://github.com/arvinth777)
- LinkedIn: [Arvinth Cinmayan](https://www.linkedin.com/in/arvinth-cinmayan)
- Portfolio: [arvinthcinmayan.com](https://arvinthcinmayan.com)

## 📧 Contact

For questions or feedback, please open an issue on GitHub or reach out via email.

---

<div align="center">
  <p>Made with ❤️ using Streamlit and TensorFlow</p>
  <p>⭐ Star this repo if you find it helpful!</p>
</div>
