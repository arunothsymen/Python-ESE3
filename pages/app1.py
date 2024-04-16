import streamlit as st
from PIL import Image

# Function to perform image processing operations
def image_processing(image, techniques):
    # Convert image to PIL format
    img = Image.open(image)

    # Resize
    if "Resize" in techniques:
        width = st.number_input("Enter width:", value=img.width)
        height = st.number_input("Enter height:", value=img.height)
        img = img.resize((int(width), int(height)))

    # Grayscale conversion
    if "Grayscale" in techniques:
        img = img.convert("L")

    # Image cropping
    if "Cropping" in techniques:
        left = st.number_input("Enter left coordinate:", value=0)
        top = st.number_input("Enter top coordinate:", value=0)
        right = st.number_input("Enter right coordinate:", value=img.width)
        bottom = st.number_input("Enter bottom coordinate:", value=img.height)
        img = img.crop((int(left), int(top), int(right), int(bottom)))

    # Image rotation
    if "Rotation" in techniques:
        angle = st.slider("Enter rotation angle:", min_value=0, max_value=360, value=0)
        img = img.rotate(angle)

    return img

def main():
    st.title("Image Processing App")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Original Image", use_column_width=True)

        # Image processing techniques selection
        techniques = st.multiselect("Select image processing techniques:", ["Resize", "Grayscale", "Cropping", "Rotation"])

        # Perform image processing
        processed_img = image_processing(uploaded_file, techniques)

        # Display processed image
        st.subheader("Processed Image")
        st.image(processed_img, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
