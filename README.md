# DigitDraw AI 🔢

**Handwritten Digit Recognition using Convolutional Neural Networks (CNN)**

DigitDraw AI is an interactive machine learning application that recognizes handwritten digits from 0 to 9. Users can draw a digit on a digital canvas and receive a prediction with confidence scores and a visualization of the model's output probabilities.

## ✨ Features

- Interactive drawing canvas
- Handwritten digit recognition from 0–9
- CNN model built with TensorFlow and Keras
- Image preprocessing and resizing to 28 × 28 pixels
- Prediction confidence and class probability visualization
- Clear canvas and draw another digit
- Interactive web interface powered by Streamlit

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** TensorFlow, Keras
- **Frontend:** Streamlit
- **Image Processing:** Pillow, NumPy
- **Data Visualization:** Plotly
- **Data Handling:** Pandas

## 🧠 How It Works

1. The user draws a digit on the canvas.
2. The application converts the drawing to grayscale.
3. The digit is cropped, resized, centered and normalized.
4. The processed image is reshaped to match the CNN input.
5. The trained model predicts the digit.
6. The application displays the predicted class and class probabilities.

## 📁 Project Structure

```text
DigitDraw-AI/
├── app.py
├── digit_cnn.keras
├── requirements.txt
├── README.md
├── .gitignore
└── assets/
    └── app-screenshot.png
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 recommended
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/DigitDraw-AI.git
cd DigitDraw-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

Make sure `digit_cnn.keras` is present in the project folder, then run:

```bash
python -m streamlit run app.py
```

Open the local URL shown in your terminal.

## 📊 Machine Learning

The application uses a Convolutional Neural Network (CNN) for image classification.

The model architecture includes convolutional layers, max-pooling, flattening, a dense layer, dropout and a final softmax output layer with 10 classes.

**Input shape:** 28 × 28 grayscale image  
**Output:** Probabilities for digits 0–9

## 🎯 Learning Outcomes

- Building and using a CNN for image classification
- Image preprocessing and normalization
- Loading a trained Keras model
- Integrating machine learning with Streamlit
- Visualizing model predictions and probabilities

## 🔮 Future Improvements

- Improve prediction accuracy on different handwriting styles
- Evaluate the model on a held-out test dataset
- Add sample digits and prediction history
- Deploy the application online
- Experiment with different CNN architectures

## 👨‍💻 Author

**Your Name**

- GitHub: https://github.com/YOUR-USERNAME

## 📄 License

This project is available under the MIT License if a `LICENSE` file is included.
