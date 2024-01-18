import streamlit as st
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image  # Import Image from PIL for image handling

# Load the saved model
model = load_model(r'D:\Pneumonia detection\pneumonia_detection_model.h5')

# Function to preprocess the image
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)  # Load the image
    img = img_to_array(img)  # Convert the image to numpy array
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img /= 255.0  # Rescale the image
    return img

# Function to provide suggestions based on prediction
def provide_suggestions(prediction_percentage):
    if prediction_percentage > 80:
        st.write(f"The image is predicted as Pneumonia with a confidence of {prediction_percentage:.2f}%.")
        st.write("\nSuggested Actions and Advice:")
        st.write("1. Consult a healthcare professional immediately for proper diagnosis and treatment.")
        st.write("2. Rest adequately and stay hydrated.")
        st.write("3. Monitor symptoms and temperature regularly.")
        st.write("4. Follow prescribed treatment plans, including taking any prescribed medications, such as antibiotics.")
        st.write("5. Practice good hygiene, including regular hand washing.")
        st.write("6. Keep away from others as much as possible to prevent spreading the infection.")
    else:
        st.write("The image is predicted as Normal.")

def main():
    st.title("Pneumonia Detection App")
    st.write("Upload a chest X-ray image to check for pneumonia.")

    uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.write("Image preview:")
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write("Processing...")

        # Preprocess the image
        preprocessed_image = preprocess_image(uploaded_file, target_size=(150, 150))

        # Make a prediction
        prediction = model.predict(preprocessed_image)

        # Convert the prediction to percentage
        prediction_percentage = prediction[0][0] * 100

        # Provide suggestions
        provide_suggestions(prediction_percentage)

if __name__ == "__main__":
    main()
   