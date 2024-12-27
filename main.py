import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import base64

def add_background(image_file):
    with open(image_file, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});  
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def add_sidebar_background(image_file):
    with open(image_file, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode()

    st.markdown(
        f"""
        <style>
        .sidebar .sidebar-content {{
            background-image: url(data:image/png;base64,{encoded_image});  
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            padding: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

background_image_path = "C:/Users/dell/Desktop/Plant_Disease_Prediction/background.jpg"  
add_background(background_image_path)

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Dataset", "Disease Recognition"])

if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes different techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Dataset
    Learn more about the dataset, on the **About Dataset** page.
    """)

elif app_mode == "About Dataset":
    st.header(" Dataset")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on [Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data).

                This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
                
                A new directory containing 33 test images is created later for prediction purposes.
                
                #### Content
                1. Train (70,295 images)
                2. Test (33 images)
                3. Validation (17,572 images)
                """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)

        class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                       'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                       'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                       'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                       'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                       'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                       'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                       'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                       'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                       'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                       'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                       'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                       'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                       'Tomato___healthy']

        predicted_class = class_names[result_index]
        st.success(f"Model is Predicting it's a {predicted_class}")

        df = pd.read_excel("C:/Users/dell/Desktop/Plant_Disease_Prediction/pest.xlsx")

        disease_info = df[df['Disease_Name'] == predicted_class]
        if not disease_info.empty:
            st.subheader("Supplemental Information:")
            for index, row in disease_info.iterrows():
                for col in disease_info.columns:
                    value = row[col]
                    if isinstance(value, str) and value.startswith("http"):
                        st.markdown(f"**{col}:** [Link]({value})")
                    else:
                        st.markdown(f"**{col}:** {value}")
        else:
            st.warning("No additional information found for this disease.")