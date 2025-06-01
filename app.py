import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import network
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Load trained model
net = network.Network.load("trained_network.pkl")

st.set_page_config(page_title="Digit Classifier", layout="centered")
st.title("Digit Classifier Neural Network")

CANVAS_SIZE = 280
DIGIT_SIZE = 28

image_result = None

st.markdown("### Input")
st.markdown("##### Upload an image or draw a digit")

# Layout: Make columns equal width for equal sized displays
left_col, right_col = st.columns([1, 1])

with right_col:
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Read the image file
        image = Image.open(uploaded_file)
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        # Resize to match canvas size
        image = image.resize((CANVAS_SIZE, CANVAS_SIZE))
        # Display the uploaded image
        with left_col:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            # Toogle inverse colors
            if st.checkbox("Invert colors"):
                image = Image.eval(image, lambda x: 255 - x)
        # Convert to numpy array
        image_result = np.array(image)
    else:
        with left_col:
            canvas_result = st_canvas(
                fill_color="#FFFFFF",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=CANVAS_SIZE,
                width=CANVAS_SIZE,
                drawing_mode="freedraw",
                key="canvas",
                update_streamlit=True,
            )
            if canvas_result.image_data is not None:
                image_result = canvas_result.image_data.astype(np.uint8)


# Helper: crop + center like MNIST
def crop_and_center(img):
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return np.zeros((28, 28))
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]
    scale = 20.0 / max(w, h)
    resized = cv2.resize(cropped, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    result = np.zeros((28, 28), dtype=np.float32)
    x_offset = (28 - resized.shape[1]) // 2
    y_offset = (28 - resized.shape[0]) // 2
    result[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
    return result / 255.0

# Prediction
if st.button("Check digit"):
    if image_result is not None:
        rgba = image_result
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

        # Preprocess
        processed = crop_and_center(gray)
        input_vector = processed.reshape(784, 1)
        output = net.feedforward(input_vector)
        prediction = int(np.argmax(output))

        st.markdown("### Processed (28×28)")
        # Convert processed image to a displayable format and resize to match canvas
        display_img = (processed * 255).astype(np.uint8)
        # Resize to match the canvas size
        display_img = cv2.resize(display_img, (CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)
        st.image(display_img, width=CANVAS_SIZE, channels="GRAY")

        # Results
        st.markdown(f"### 🔍 Prediction: **{prediction}**")
        # Use a specific height for the bar chart to keep it compact
        st.bar_chart(output.flatten(), height=200)
    else:
        st.warning("Please draw a digit first.")
