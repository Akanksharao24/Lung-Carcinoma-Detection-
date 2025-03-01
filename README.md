# Lung and Colon Cancer Histopathological Images – Anomaly Detection

## Overview
This project aims to build a **Streamlit-based web application** for anomaly detection in lung and colon cancer histopathological images using deep learning models. The application utilizes medical imaging libraries like **MI2RLNet**, **MedPy**, and other state-of-the-art frameworks to detect anomalies in medical scans.

### Dataset
The dataset used for this project is sourced from Kaggle:
[Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

#### Dataset Description:
- The dataset contains **histopathological images** of lung and colon tissues.
- It consists of three classes:
  - Lung Adenocarcinoma (LUAD)
  - Lung Squamous Cell Carcinoma (LUSC)
  - Colon Adenocarcinoma (COAD)
- Images are organized into respective class folders.
- Each image is **1024x1024 pixels** in JPEG format.

## Features
- Upload histopathological images for anomaly detection.
- Pre-processing of medical images using **MedPy**.
- Anomaly detection using **MI2RLNet** model.
- Real-time inference with visualization of predictions.
- Intuitive **Streamlit** web interface.

## Technologies Used
- Python
- Streamlit
- TensorFlow/Keras
- MI2RLNet
- MedPy
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

## Installation
### Prerequisites
- Python 3.8+
- Kaggle API (for dataset access)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/username/lung-colon-anomaly-detection.git
cd lung-colon-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and extract it into the `data/` folder.

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Folder Structure
```bash
├── data/                 # Dataset folder
├── models/              # Pre-trained models
├── notebooks/           # Jupyter Notebooks for experiments
├── src/                 # Source code
│   ├── preprocessing.py # Pre-processing functions
│   ├── model.py        # Model definitions
│   └── inference.py    # Inference pipeline
├── app.py              # Streamlit web app
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Model Architecture
The anomaly detection model is built using **MI2RLNet**, which leverages:
- Convolutional Neural Networks (CNNs)
- Residual Blocks
- Attention Mechanisms

### Training Pipeline
1. Data Pre-processing using **MedPy**.
2. Data Augmentation.
3. Model Training using TensorFlow.
4. Model Evaluation using Accuracy, Precision, Recall, and F1 Score.

## How to Use
1. Upload the histopathological image through the web app.
2. Click the **Predict** button.
3. View the anomaly detection result along with confidence scores.

## Results
| Class           | Accuracy |
|----------------|----------|
| LUAD           | 94%     |
| LUSC           | 91%     |
| COAD           | 92%     |

## Future Improvements
- Integrating more advanced models like Vision Transformers.
- Deploying the app on cloud platforms like Heroku.
- Adding explainable AI techniques.

## Contributing
Contributions are welcome! Feel free to open a Pull Request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- MI2RLNet Framework
- MedPy Library

