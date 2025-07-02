import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
model = load_model("sign_language_model.keras")
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
if len(class_labels) == 26:
    class_labels.remove('J')
    class_labels.remove('Z')  
st.title(" AI Sign Language Translator")
st.write("Upload an image of a hand sign (28x28 grayscale) and get its predicted letter.")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  
    st.image(image, caption="Uploaded Image", width=150)
    image = ImageOps.fit(image, (28, 28), Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    st.write(f"### Predicted Letter: *{class_labels[predicted_class]}*")
