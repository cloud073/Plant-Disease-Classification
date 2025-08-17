# Plant Disease Classification using ResNet-9
>**This project enables real-time plant leaf disease classification in 10–20 seconds per image using a lightweight ResNet-9 deep learning model. The app and backend are fully integrated with [Hugging Face]([https://huggingface.co](https://huggingface.co/teamtezstory))—making rapid, reliable predictions possible from both mobile devices and web interfaces.**

![Summer Plant Care Tips](https://yummycake.in/wp-content/uploads/2025/06/summer-plant-care-tips.jpg)

---

## Features
- **High-Accuracy Model:** 97.7% verified on the PlantVillage dataset.
- **FastAPI/Hugging Face Deployment:** REST API and web interface hosted via Hugging Face Spaces for easy, scalable access.
- **Android App:** Instant leaf health prediction with the included `plantcare.apk`—using the Hugging Face API as backend.
- **Comprehensive EDA:** Jupyter Notebook provided for dataset analysis and model development.
- **Simple Integration:** REST API is accessible from browser, scripts, and mobile app.

---

## 🚀 Hugging Face Demo Spaces

**Try the live demos now:**

### 1. Start `Plantcare` Space
- [Plantcare Space](https://huggingface.co/spaces/teamtezstory/plantcare)
- Identify plant diseases directly from browser via image upload.
- Powered by FastAPI and PyTorch backend deployed on Hugging Face.

### 2. Start `Plantcareapp` Space
- [Plantcareapp Space](https://huggingface.co/spaces/teamtezstory/plantcareapp)
- Mobile-friendly interface for quick leaf disease prediction.
- The Android app can connect to this space for instant results.

**How to test with your data:**
- Use any test photo from your `/data/test` folder.
- Upload the image to either Space and view prediction results (plant name, disease type, confidence).

## How the Hugging Face Deployment Works

- **API Endpoint**: The trained ResNet-9 model is exported and served via a FastAPI application deployed on Hugging Face Spaces.
- **Android App**: Uses the Hugging Face Space's REST endpoint for prediction—simply install, capture leaf image, and get results.
- **Web Demo**: Optionally, use the Hugging Face Space's GUI for browser-based predictions.

---

## Dataset Organization

- **Source:** [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Folders:**
```plaintext
data/
├── train/   # Training images (80%)
├── valid/   # Validation images (20%)
├── test/    # Real-world prediction set
└── readme   # Dataset notes
```
- **Classes:** 38 (14 plants, 26 diseases)
- **Preprocessing:** 256x256, augmented with flips, rotations, color jitter.

---

## Technologies

- **Python 3.8+**
- **PyTorch**
- **FastAPI**
- **Hugging Face Spaces**
- **Android Studio** (`app/plantcare.apk`)
- **Libraries:** torch, torchvision, PIL, matplotlib, pandas, numpy

---

## Usage Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/cloud073/Plant-Disease-Classification
cd plant-disease-classification
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
Place leaf images in the correct folders under `data/`.

### 4. Train/Load Model
Use `plant-disease-classification.ipynb` for training. Export model to `.pkl` or Hugging Face format.

### 5. Deploy on Hugging Face Spaces
- Push FastAPI code and model weights to a Hugging Face Space.

### 6. Android & API Integration
- Set Hugging Face Space API endpoint in the app's configuration or code.
- For prediction:
  - Capture/choose leaf image in Android app.
  - The app sends the image to Hugging Face API.
  - Prediction (plant species, disease type, confidence) returns within 10–20sec.
---

## Model Details

- **Architecture:** ResNet-9, optimized for speed and edge/online deployment.
- **Input:** RGB leaf images (256x256)
- **Output:** Predicted plant, disease, confidence score (<20sec inference).

---

## Acknowledgments
- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- Hugging Face team and Spaces platform
- FastAPI, PyTorch, and Android developer communities

