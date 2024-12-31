from PIL import Image, ImageDraw, ImageFont
import random
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2

# Function to generate an image of a random letter (A-Z)
def generate_letter_image(letter):
    # Create a blank white image (28x28 pixels)
    img = Image.new('L', (28, 28), color=255)  # 'L' mode is grayscale
    draw = ImageDraw.Draw(img)
    
    # Use a built-in font
    try:
        font = ImageFont.truetype("arial.ttf", 24)  # Use a font size that fits well
    except IOError:
        font = ImageFont.load_default()
    
    # Get the bounding box of the letter to center it
    bbox = draw.textbbox((0, 0), letter, font=font)  # Get the bounding box of the text
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]  # Calculate width and height of the text
    position = ((28 - width) // 2, (28 - height) // 2)  # Center the text
    
    # Draw the letter on the image
    draw.text(position, letter, fill=0, font=font)
    
    # Convert the image to a numpy array and return it
    img = np.array(img)
    return img.flatten()

# Function to generate a dataset of random letters
def generate_dataset(num_samples=1000):
    data = []
    labels = []
    
    for _ in range(num_samples):
        letter = random.choice(string.ascii_uppercase)  # Random letter A-Z
        img = generate_letter_image(letter)
        data.append(img)
        labels.append(letter)
    
    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# Generate a dataset of random letters
data, labels = generate_dataset(1000)

# Preprocess the data: Normalize the pixel values to 0-1
data = data.astype(np.float32) / 255.0  # Normalize pixel values to 0-1

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a Support Vector Machine (SVM) classifier
svm = SVC(kernel='linear')  # Linear kernel for simplicity
svm.fit(x_train, y_train)

# Evaluate the model
y_pred = svm.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Predict a single letter
def predict_letter(image, model):
    image = image.reshape(1, -1)  # Flatten the image
    prediction = model.predict(image)
    return prediction[0]

# Test on a single letter
sample_image = x_test[0].reshape(28, 28)  # Reshape the image to 28x28 for visualization
cv2.imshow("Sample Image", sample_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

prediction = predict_letter(sample_image.flatten(), svm)
print(f"Predicted Letter: {prediction}")