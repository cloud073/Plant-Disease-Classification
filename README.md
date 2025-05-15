# Plant Disease Classification using ResNet-9

## 📌 Overview
This project focuses on classifying plant diseases using a deep learning model based on ResNet-9. The goal is to distinguish between healthy and diseased crop leaves and, if diseased, identify the specific disease. The model is trained on an augmented dataset derived from the original PlantVillage Dataset, containing about 87K RGB images categorized into 38 classes.

## 🚀 Features
- **Dataset**: The dataset includes 87K RGB images of healthy and diseased crop leaves, split into 80% training and 20% validation sets. An additional directory contains 33 test images for prediction.
- **Model Architecture**: Utilizes ResNet-9, a lightweight variant of the ResNet architecture, optimized for efficient training and inference.
- **Deployment**: The model is deployed using FastAPI for backend services and integrated into an Android app for mobile access.
- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of the dataset, including visualizations of class distributions and unique plant/disease counts.

## 📊 Dataset Description
- **Source**: Derived from the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset).
- **Classes**: 38 unique classes (14 plants and 26 diseases, excluding healthy leaves).
- **Splits**: 80% training, 20% validation, and a separate test set.
- **Augmentation**: Offline augmentation applied to enhance dataset diversity.

## 🛠️ Technologies Used
- **Python**: Primary programming language.
- **PyTorch**: Deep learning framework for model training.
- **FastAPI**: Backend framework for deploying the model.
- **Android Studio**: Used to develop the mobile application.
- **Libraries**:
  - `torch`, `torchvision` for deep learning.
  - `PIL`, `matplotlib` for image processing and visualization.
  - `pandas`, `numpy` for data manipulation.

## 📂 Project Structure
1. **Data Exploration**:
   - Load and analyze the dataset.
   - Visualize class distributions and unique plant/disease counts.
2. **Model Training**:
   - Implement ResNet-9 architecture.
   - Train the model on the augmented dataset.
3. **FastAPI Deployment**:
   - Set up a REST API to serve the model.
   - Handle image uploads and return predictions.
4. **Android App**:
   - Develop an app to capture/upload leaf images.
   - Display disease classification results.

## 🏆 Results
- Achieved high accuracy in classifying plant diseases.
- Successfully deployed the model for real-time predictions via FastAPI and Android app.

## 📝 How to Use
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd plant-disease-classification
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the FastAPI Server**:
   ```bash
   uvicorn main:app --reload
   ```
4. **Android App**:
   - Open the project in Android Studio.
   - Build and run the app on an emulator or device.

## 🌟 Future Enhancements
- Expand the dataset to include more plant species and diseases.
- Optimize the model for edge devices to reduce inference time.
- Add multilingual support in the Android app.

## 🙏 Acknowledgments
- Thanks to the creators of the PlantVillage Dataset for providing the foundational data.
- The FastAPI and Android communities for their excellent documentation and support.

