import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model with custom objects handling
MODEL_PATH = "efficientnetb1_model.h5"

def load_model(model_path):
    try:
        from tensorflow.keras.layers import BatchNormalization
        model = tf.keras.models.load_model(model_path, custom_objects={"BatchNormalization": BatchNormalization})
        return model
    except Exception as e:
        return f"Error loading model: {e}"

model = load_model(MODEL_PATH)

# Define image preprocessing function
def preprocess_image(image):
    image = image.resize((300, 300))  # Resize to EfficientNetB3 input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define prediction function
def predict(image):
    if isinstance(model, str):
        return model  # If model loading failed, return error message
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    
    # Convert to heatmap for carcinoma detection visualization
    heatmap = np.uint8(255 * processed_image[0])  # Scale image to 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply heatmap
    heatmap = Image.fromarray(heatmap)  # Convert back to PIL Image
    
    if prediction > 0.5:
        return "Lung Carcinoma Detected! Consult a doctor immediately.", heatmap
    else:
        return "No Lung Carcinoma detected.", heatmap

# Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(), gr.Image(type="pil")],
    title="Lung Carcinoma Detection",
    description="Drop a chest X-ray or CT scan to check for lung carcinoma. The heatmap indicates possible carcinoma regions."
)

demo.launch()
