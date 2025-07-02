import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
model = load_model("sign_language_model.keras")

# Define class labels (adjust if you have 24/25/26)
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
if len(class_labels) == 26:
    class_labels.remove('J')
    class_labels.remove('Z')  # since Sign Language MNIST doesn't include J and Z

st.title(" AI Sign Language Translator")
st.write("Upload an image of a hand sign (28x28 grayscale) and get its predicted letter.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded Image", width=150)
    
    # Resize to 28x28 and preprocess
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Show result
    st.write(f"### Predicted Letter: *{class_labels[predicted_class]}*")
