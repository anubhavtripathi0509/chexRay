import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the pre-trained VGG16 model
model = load_model('model_vgg16.h5')

# Function to preprocess the uploaded image
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict(image_file):
    img_array = preprocess_image(image_file)
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
def main():
    st.title("Chest X-Ray Image Classifier")
    st.write("Upload a chest X-ray image to classify whether it's normal or pneumonia.")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Make predictions when the user clicks the button
        if st.button("Classify"):
            with st.spinner('Classifying...'):
                # Predict the class of the uploaded image
                prediction = predict(uploaded_file)

            # Display the result
            if prediction[0][0] > 0.5:
                st.error("Prediction: Normal")
            else:
                st.success("Prediction: Pneumonia")

if __name__ == '__main__':
    main()
