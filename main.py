import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model('efficientnet_plantvillage2.h5')

    # Load image from uploaded file (buffer)
    image = Image.open(test_image).convert("RGB")
    image = image.resize((100, 100))

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57;'>
        🌿 Plant Disease Recognition System
    </h1>
    <p style='text-align: center; font-size: 18px;'>
        Upload a plant leaf image and let the AI analyze its health.
    </p>
    """,
    unsafe_allow_html=True,
)

st.write("---")

st.subheader("🔍 Upload Leaf Image")

uploaded_file = st.file_uploader(
    "Choose an image of a plant leaf",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.info("Click **Predict** to analyze the plant leaf.")
        predict_btn = st.button("🔎 Predict Disease", use_container_width=True)

        if predict_btn:
            with st.spinner("Analyzing image... Please wait..."):
                result_index = model_prediction(uploaded_file)

                # class list
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]

                pred_label = class_name[result_index]

                st.success("Prediction Complete")

                st.markdown(
                    f"""
                    <div style="
                        padding: 20px;
                        border-radius: 10px;
                        background-color: #F0FFF0;
                        border-left: 6px solid #2E8B57;
                        margin-top: 20px;
                    ">
                        <h3 style="color:#2E8B57;">🌱 Result: <b>{pred_label}</b></h3>
                        <p style="font-size:17px;">
                            The system predicts that this leaf belongs to:
                            <b>{pred_label}</b>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
