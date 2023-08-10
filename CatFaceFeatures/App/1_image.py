from sys import version
import cv2
import tensorflow as tf
import numpy as np
import gradio as gr

# Load the trained model from the saved file
loaded_model = tf.keras.models.load_model('CatFaceFeatures_Resnet50_2.h5')

# Function to predict facial landmarks on new images
def predict_landmarks(image_input):
    # Convert Gradio image object to numpy array
    image = image_input.astype('uint8')

    # Define the image size for resizing
    image_size = (224, 224)

    # Convert to RGB before resizing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, image_size)
    input_image = np.expand_dims(resized_image, axis=0)

    # Make predictions using the trained model
    predictions = loaded_model.predict(input_image)

    # Rescale the predictions to the original image size
    scale_y = image.shape[0] / image_size[0]
    scale_x = image.shape[1] / image_size[1]
    resized_predictions = [int(value * scale_x) if i % 2 == 0 else int(
        value * scale_y) for i, value in enumerate(predictions[0])]

    # Calculate the radius of the circles based on image dimensions
    image_height, image_width, _ = image.shape
    max_dim = max(image_height, image_width)
    radius_scale = max_dim / 1500  # Adjust this scale factor as needed
    
    # Draw circles (dots) on the original image at the predicted landmark locations
    for i in range(0, len(resized_predictions), 2):
        x, y = resized_predictions[i], resized_predictions[i + 1]
        color = (255, 0, 0)
        radius = int(8 * radius_scale)  # Adjust the base radius value as needed
        thickness = -1
        cv2.circle(image, (x, y), radius, color, thickness)
        
    return image

# Define the Gradio input component
image_input = gr.inputs.Image()

# Create the Gradio interface
gr.Interface(fn=predict_landmarks, inputs=image_input, share=True, outputs="image").launch()



