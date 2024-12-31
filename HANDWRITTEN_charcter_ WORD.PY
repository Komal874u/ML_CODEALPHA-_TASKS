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
    # Create a blank white image (56x56 pixels) for better detail
    img = Image.new('L', (56, 56), color=255)  # 'L' mode is grayscale
    draw = ImageDraw.Draw(img)
    
    # Use a built-in font
    try:
        font = ImageFont.truetype("arial.ttf", 40)  # Use a larger font size
    except IOError:
        font = ImageFont.load_default()
    
    # Get the bounding box of the letter to center it
    bbox = draw.textbbox((0, 0), letter, font=font)  # Get the bounding box of the text
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]  # Calculate width and height of the text
    position = ((56 - width) // 2, (56 - height) // 2)  # Center the text
    
    # Draw the letter on the image
    draw.text(position, letter, fill=0, font=font)
    
    # Convert the image to a numpy array and return it
    img = np.array(img)
    return img.flatten()

# Function to generate an image of a word
def generate_word_image(word):
    # Each letter will be 56x56 pixels, and we need to space them
    width = len(word) * 56  # Adjusted image size
    img = Image.new('L', (width, 56), color=255)  # Create a white background image
    
    # Create a drawing object
    draw = ImageDraw.Draw(img)
    
    # Use a built-in font
    try:
        font = ImageFont.truetype("arial.ttf", 40)  # Use a larger font size
    except IOError:
        font = ImageFont.load_default()

    # Draw each letter in the word
    for i, letter in enumerate(word):
        draw.text((i * 56, 0), letter, fill=0, font=font)  # Draw each letter at the correct position

    # Convert the image to a numpy array and return it
    img = np.array(img)
    return img.flatten()

# Function to generate a dataset of random words
def generate_dataset(num_samples=1000, word_length=5):
    data = []
    labels = []
    
    for _ in range(num_samples):
        word = ''.join(random.choices(string.ascii_uppercase, k=word_length))  # Random word
        img = generate_word_image(word)
        data.append(img)
        labels.append(word)
    
    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# Generate a dataset of random words
data, labels = generate_dataset(1000, word_length=5)

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

# Predict a single word
def predict_word(image, model):
    image = image.reshape(1, -1)  # Flatten the image
    prediction = model.predict(image)
    return prediction[0]

# Test on a single word
sample_image = x_test[0].reshape(56, len(y_test[0]) * 56)  # Reshape the image for word length
cv2.imshow("Sample Image", sample_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

prediction = predict_word(sample_image.flatten(), svm)
print(f"Predicted Word: {prediction}")