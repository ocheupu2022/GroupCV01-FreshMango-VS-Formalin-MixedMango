# Fruit Quality Classifier

A deep learning web app that classifies fruit images as formalin-mixed or fresh. Built with MobileNetV3 transfer learning on TensorFlow/Keras and deployed via Streamlit.

---

## Demo

Upload any fruit image and the model returns the predicted quality class with confidence scores.

---

## Repository Structure

```
fruit-quality-classifier/
├── mobilenetv3_transfer.keras   # saved trained model
├── app.py                       # Streamlit web app
├── requirements.txt             # Python dependencies
├── .python-version              # Python 3.11
└── README.md
```

---

## Model

- Architecture: MobileNetV3Small (transfer learning, frozen base)
- Input shape: `(224, 224, 3)`
- Classes: `formalin_mixed`, `fresh`
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## Dataset

[Fruits Disease Dataset — Kaggle](https://www.kaggle.com/datasets/saravanansri/fruits-disease-dataset)

```python
import kagglehub

path = kagglehub.dataset_download("saravanansri/fruits-disease-dataset")
print("Path to dataset files:", path)
```

Dataset path on Kaggle: `/kaggle/input/datasets/saravanansri/fruits-disease-dataset`


---

## Run Locally

1. Clone the repo

```bash
git clone https://github.com/<your-repo-username>/<your-repo-name>.git
cd <your-repo-name>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run app.py
```

---

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- MobileNetV3Small (ImageNet weights)
- Streamlit
- NumPy
- Pillow

---

