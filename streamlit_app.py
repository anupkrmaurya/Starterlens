import streamlit as st
import torch
import pickle
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import requests

# --- CNN model definition ---
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dense_layers(x)
        return x

# --- Constants ---
class_names = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
NUMBER_OF_CLASSES = len(class_names)

# --- Load CNN model ---
with open("eye_detection_model.pkl", "rb") as f:
    model = pickle.load(f)
model.eval()

# --- Image preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --- OpenRouter API ---
API_KEY = "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"
MODEL_NAME = "google/gemini-2.0-flash-001"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def get_short_llm_response(prompt, max_tokens=150):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    try:
        return result["choices"][0]["message"]["content"].strip()
    except:
        return result["choices"][0]["content"].strip()

# --- Streamlit UI ---

# Sidebar info
st.sidebar.success("Model loaded successfully!")
# st.sidebar.info(f"Model type: CNN")
st.sidebar.info(f'Created by Team: StarterLens')
st.sidebar.markdown(
    """
    <div style="background-color: rgba(255, 0, 0, 0.7); 
                padding: 10px; 
                border-radius: 5px; 
                color: white; 
                ">
        Members: Anoop Maurya, Yashwant Kumar, Abhisek Kumar
    </div>
    """,
    unsafe_allow_html=True
)



# App title and instructions
st.markdown("<h1 style='text-align: Center;'>StarterLens AI ~ Eye Detection</h1>", unsafe_allow_html=True)
# st.markdown("<h5 style='text-align: Center;'> Application</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("👁️Upload an image to detect eyes using your trained model")

# Upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
diagnosis_text = ""
predicted_class = None
user_problem = ""

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### 📷 Original Image")
    st.image(image, use_column_width=True)

    # Predict
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, prediction = torch.max(outputs, 1)
        predicted_class = class_names[prediction.item()]
        diagnosis_text = f"The predicted eye disease is: {predicted_class.replace('_', ' ').title()}."
        st.success("✅ Prediction completed!")
        st.markdown("**Prediction Result:**")
        if predicted_class == "normal":
            st.markdown("🟢 No Eyes Detected")
        else:
            st.markdown(f"🔴 **{predicted_class.replace('_', ' ').title()} Detected**")

st.markdown("---")

user_problem = st.text_area("Describe any eye problem or symptom related to this image:")

# Initialize or reset chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Submit problem button
if st.button("Submit Problem"):
    problem = user_problem.strip()
    if not problem:
        st.warning("Please enter a problem description before submitting.")
    elif not predicted_class:
        st.warning("Please upload an image and get a prediction before submitting your problem.")
    else:
        st.session_state.chat_history = []
        # Compose prompts with diagnosis + user problem
        prompts = {
            "Why is this happening?": f"{diagnosis_text} Explain shortly why this is happening: {problem}",
            "What should you do or avoid?": f"{diagnosis_text} Briefly what should the user do or avoid: {problem}",
            "Short-term relief suggestions": f"{diagnosis_text} Give short suggestions for short-term relief for: {problem}",
            "Home remedies": f"{diagnosis_text} Suggest brief home remedies for: {problem}"
        }
        with st.spinner("Generating AI expert advice..."):
            for title, prompt in prompts.items():
                answer = get_short_llm_response(prompt, max_tokens=150)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "title": title,
                    "content": answer
                })

# Display expert assistant answers in collapse boxes
if st.session_state.chat_history:
    for entry in st.session_state.chat_history:
        with st.expander(entry["title"]):
            st.write(entry["content"])

    followup = st.text_input("Ask a follow-up question related to your eye problem:")
    if st.button("Send Follow-up") and followup.strip():
        if not predicted_class:
            st.warning("Please upload an image, get a prediction, and submit a problem before asking follow-ups.")
        else:
            # Build context including diagnosis, problem, chat history, user followup question
            context = f"{diagnosis_text}\nUser problem: {user_problem}\nConversation history:\n"
            for msg in st.session_state.chat_history:
                context += f"{msg['title']}: {msg['content']}\n"
            context += "User: " + followup.strip()

            with st.spinner("Getting follow-up response..."):
                followup_answer = get_short_llm_response(context, max_tokens=250)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "title": "Follow-up",
                    "content": followup_answer
                })
                st.rerun()

# Info sections in collapsible panels
with st.expander("📖 How to Use"):
    st.write("""
    1. Upload a retinal image file (jpg, png, etc.).
    2. Click 'Detect Eyes' to get the model's prediction.
    3. Describe any symptoms or problems related to the image.
    4. Submit the problem to get expert advice.
    5. Ask follow-up questions for tailored guidance.
    """)

with st.expander("ℹ️ Model Information"):
    st.write("""
    - 3-layer CNN trained on four classes (cataract, diabetic retinopathy, glaucoma, normal).
    - Expert advice generated via OpenRouter Gemini LLM.
    """)

# Footer
st.markdown("<center>Created By Team StarterLens❤️ </center>", unsafe_allow_html=True)
