# 👤 Age and Gender Detection + Image Captioning (BLIP)

A complete solution for predicting the age and gender of a person from an image using a CNN model, and generating a description using BLIP (Bootstrapped Language-Image Pretraining). This project also features a Gradio web application for seamless interaction with the model.

## ✨ Features
- Age and gender prediction using a CNN model 🧑‍🦳👩‍🦳  
- BLIP-based image captioning for descriptive outputs 📝  
- Gradio web app for easy image input and output display 🌐  
- Full pipeline for model training, integration, and deployment 🔄  
- Interactive web interface with easy-to-understand results 🖱️  

## 🧰 Tech Stack
- **Frontend**: Gradio (for web app deployment)  
- **Backend**: Python, TensorFlow  
- **Dataset**: UTKFace dataset  
- **Image Captioning**: BLIP (Bootstrapped Language-Image Pretraining)  
- **Model**: CNN (Convolutional Neural Network)  
- **Others**: NumPy, OpenCV, Matplotlib for data preprocessing and visualization  

## 📁 Folder Structure
```bash
age-gender-detection-and-captioning/
├── models/
│   └── age_gender_model.h5  # Trained CNN model
├── notebook/
│   └── age_gender_detection_model.ipynb  # Full training, inference, and deployment notebook
├── saved_outputs/
│   └── example_outputs/  # Sample outputs (age, gender, caption)
├── README.md  # Instructions for downloading and placing UTKFace dataset   
└── requirements.txt  # Required dependencies

```

## ▶️ How to Run

### 1. Install dependencies

Navigate to the project directory and install the necessary libraries:

```bash
pip install -r requirements.txt
```

### 2. Download the UTKFace dataset

The UTKFace dataset is not included in this repository. You can download it from Kaggle using the following command:

```bash
kaggle datasets download -d jangedoo/utkface
```

Alternatively, download it manually from [Kaggle - UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface).

Once downloaded, unzip the dataset and place it inside the `data/` directory.

### 3. Train the Model

To train the CNN model for age and gender detection:

- Open the Jupyter notebook located at `notebook/age_gender_detection_model.ipynb`
- Run the notebook cells in order:
  - Load and preprocess the UTKFace dataset
  - Define and compile the CNN model
  - Train the model on the dataset
  - Save the trained model to the `models/` directory

### 4. Generate Captions with BLIP

To generate captions using BLIP:

- Ensure the BLIP model files are downloaded and set up correctly (you can use HuggingFace’s `blip` package)
- Use the integration cells inside the same notebook to:
  - Load BLIP
  - Generate captions for the uploaded or test images

### 5. Run the Gradio Web App

To launch the Gradio interface for testing:

```python
import gradio as gr

def predict_age_gender_and_caption(image):
    age, gender = model.predict(image)
    caption = blip_model.generate_caption(image)
    return age, gender, caption

gr.Interface(fn=predict_age_gender_and_caption, inputs="image", outputs=["text", "text", "text"]).launch()
```

This interface allows users to upload an image and see:
- Predicted age
- Predicted gender
- Generated caption from BLIP

### 6. View Sample Outputs

Visit the `saved_outputs/example_outputs/` directory to preview example results generated from the trained model and BLIP captioning.

---

✅ Now you're all set to use a fully functional and intelligent age, gender, and description predictor with an interactive frontend!
