# 👁️ Eye Diseases Classification

### Deep Learning-based Eye Disease Detection & AI Assistant

This project provides an end-to-end workflow for retinal disease detection using a CNN in PyTorch. It also includes an expert advice assistant powered by OpenRouter Gemini LLM, accessible through a user-friendly Streamlit web application.

---

## Features

- **Multi-class Eye Disease Detection:**  
  Classifies retinal images as **Normal, Cataract, Diabetic Retinopathy, or Glaucoma**.

- **Interactive Web Application:**  
  Upload an image, detect disease, and get personalized advice using [Streamlit](https://streamlit.io).

- **Expert Assistant:**  
  Ask queries and get instant, contextual medical guidance via Gemini LLM, integrated with your diagnosis.

- **Easy Extensibility:**  
  Ready for new data, further model training, or other LLM integrations.

---

## Getting Started

1. **Clone the Repo**
    ```
    git clone https://github.com/your-username/eye-diseases-classification.git
    cd eye-diseases-classification
    ```

2. **Install Requirements**
    ```
    pip install -r requirements.txt
    ```
    *You may need:*
    - `torch`
    - `torchvision`
    - `streamlit`
    - `pillow`
    - `requests`

3. **Train or Download Model**
    - Place your trained model as `eye_detection_model.pkl` in the root directory.

4. **Run the App**
    ```
    streamlit run streamlit_app.py
    ```

---

## Usage

- Upload a retinal JPG or PNG image.
- Click "Detect Eyes" to get the classification result.
- Enter any symptoms or questions for the AI expert assistant.
- View answers in separate panels and ask follow-up questions.

---

## Files and Structure

