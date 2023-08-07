import cv2
import tensorflow as tf
import numpy as np

# Load the trained model from the saved file
loaded_model = tf.keras.models.load_model('CatFaceFeatures_Resnet50.h5')

# Function to predict facial landmarks on new images


def predict_landmarks(image_path):
    # Define the image size for resizing
    image_size = (224, 224)

    # Load the image and preprocess it
    image = cv2.imread(image_path)
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

    return image, resized_predictions


# Use the loaded model for predictions on a new image
new_image_path = r'C:\Users\haris\test-installation\Data\Face Mask\download.jpg'
original_image, landmarks = predict_landmarks(new_image_path)
print("Predicted Landmarks:", landmarks)

# Draw circles (dots) on the original image at the predicted landmark locations
for i in range(0, len(landmarks), 2):
    x, y = landmarks[i], landmarks[i + 1]
    # Red color for the dots, you can change it to any desired color
    color = (0, 0, 255)
    radius = 3  # Adjust the size of the dots as needed
    thickness = -1  # Fill the circles (dots) to make them solid
    cv2.circle(original_image, (x, y), radius, color, thickness)

# Show the image with predicted landmarks
cv2.imshow('Predicted Landmarks', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
