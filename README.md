# DigitDraw AI 🔢

**Handwritten Digit Recognition using Convolutional Neural Networks (CNN)**

DigitDraw AI is an interactive machine learning application that recognizes handwritten digits from **0 to 9**. Draw a digit on the canvas and receive a prediction with confidence scores and a visualization of the model's class probabilities.

## 🚀 Live Demo

**Try the app:** [DigitDraw AI](https://digitdraw-ai.streamlit.app/)

## ✨ Features

- Interactive drawing canvas
- Handwritten digit recognition from 0–9
- CNN model built with TensorFlow and Keras
- Image preprocessing: grayscale conversion, cropping, resizing, centering, and normalization
- Prediction confidence and class probability visualization
- Clear the canvas and draw another digit
- Interactive web interface powered by Streamlit

## 🛠️ Tech Stack

- **Language:** Python
- **Machine Learning:** TensorFlow, Keras
- **App UI:** Streamlit
- **Image Processing:** Pillow, NumPy
- **Data Handling:** Pandas
- **Visualization:** Plotly

## 🧠 How It Works

1. Draw a digit on the canvas.
2. The app converts the drawing to grayscale.
3. The digit is cropped, resized to fit a 28 × 28 image, centered, and normalized.
4. The processed image is passed to the trained CNN.
5. The model predicts a digit from 0 to 9.
6. The app displays the prediction and class probabilities.

## 📁 Project Structure

```text
DigitDraw-AI/
├── app.py
├── digit_cnn.keras
├── requirements.txt
├── README.md
└── .gitignore
```

Keep `digit_cnn.keras` in the same directory as `app.py`, because the application loads the trained model from the project folder.

## ⚙️ Run Locally

### Prerequisites

- Python 3.11 recommended
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/abhiii04-25/DigitDraw-AI.git
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

### 4. Run the app

Make sure `digit_cnn.keras` is present in the project folder, then run:

```bash
python -m streamlit run app.py
```

Open the local URL printed in your terminal.

## 🧩 Model Overview

The application uses a Convolutional Neural Network (CNN) with convolutional and max-pooling layers, followed by flattening, a dense layer, dropout, and a softmax output layer for 10 digit classes.

- **Input:** 28 × 28 grayscale image
- **Output:** Class probabilities for digits 0–9

Training accuracy alone does not measure performance on unseen handwriting. Evaluate the model on a separate labeled test dataset before reporting generalization accuracy.

## 🎯 Learning Outcomes

- Building and using a CNN for image classification
- Preprocessing and normalizing images for model input
- Loading a trained Keras model for inference
- Integrating a machine learning model with Streamlit
- Visualizing predictions and class probabilities

## 🔮 Future Improvements

- Evaluate performance on a held-out test dataset
- Improve predictions across different handwriting styles
- Add sample digits and prediction history
- Add an application screenshot or demo GIF
- Experiment with alternative CNN architectures

## 👨‍💻 Author

**Abhishek Sontakke**

- GitHub: [abhiii04-25](https://github.com/abhiii04-25)

## 📄 License

No license is specified yet. Add a `LICENSE` file if you choose to publish this project under an open-source license.
