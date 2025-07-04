import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load MNIST data
(train_images, _), (test_images, _) = mnist.load_data()

# Pre-processing (normalize pixel values between 0 and 1)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape to include channels (assuming grayscale images)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)
# Define DAE model (replace with your specific architecture)
def create_dae_model():
    inputs = keras.Input(shape=(28, 28, 1))
    encoded = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    decoded = keras.layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(encoded)
    decoded = keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
    return keras.Model(inputs=inputs, outputs=decoded)

# Create and compile DAE model
model = create_dae_model()
model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the DAE model (adjust epochs and batch size)
n_epochs = 30
batch_size = 64
model.fit(train_images, train_images, epochs=n_epochs, batch_size=batch_size)

# Save the trained model
model.save('mnist_dae_model.h5')
model = keras.models.load_model('mnist_dae_model.h5')


# Get reconstructions for test data
test_reconstructions = model.predict(test_images)

# Calculate reconstruction error (mean squared error)
reconstruction_error = np.mean(np.square(test_images - test_reconstructions), axis=(1, 2, 3))

# Define anomaly threshold based on your data analysis (e.g., 95th percentile)
threshold = np.quantile(reconstruction_error, 0.95)

# Identify anomalies based on reconstruction error exceeding threshold
anomaly_indices = np.where(reconstruction_error > threshold)[0]

# Get three random anomaly indices
anomaly_indices_to_display = np.random.choice(anomaly_indices, size=3, replace=False)

# Display and save anomaly images
for i, idx in enumerate(anomaly_indices_to_display):
    plt.figure()
    plt.imshow(test_images[idx].reshape(28, 28), cmap='gray')
    plt.title(f'Anomaly Image {i+1}')
    plt.axis('off')
    plt.savefig(f'anomaly_image_{i+1}.jpg', bbox_inches='tight')

plt.show()

Video_Anomaly_Detection.ipynb
import cv2
import numpy as np
import random
import tkinter as tk
from tkinter import filedialog

def select_video_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi")])
    return file_path

def find_anomalous_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        return
    
    # Initialize previous centroid
    prev_centroid = None
    
    # Create background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    
    anomalous_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fgmask = bg_subtractor.apply(gray)
        
        # Apply morphological operations to remove noise
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((20,20),np.uint8))
        
        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes and calculate centroids
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (x + w // 2, y + h // 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
            
            # Check direction of motion
            if prev_centroid:
                dx = centroid[0] - prev_centroid[0]
                dy = centroid[1] - prev_centroid[1]
                if dx * dy < 0:  # Opposite direction motion
                    anomalous_frames.append(frame)
            
            # Update previous centroid
            prev_centroid = centroid
        
        # Display frame
        cv2.imshow("Anomaly Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if anomalous_frames:
        random_frame = random.choice(anomalous_frames)
        cv2.imshow("Random Anomalous Frame", random_frame)
        cv2.imwrite("anomaly_frame.jpg", random_frame)
        print("Random anomalous frame saved as 'anomaly_frame.jpg'")
        cv2.waitKey(0)
    else:
        print("No anomalous frames found")

if __name__ == "__main__":
    video_path = select_video_file()
    if video_path:
        find_anomalous_frame(video_path)
    else:
        print("No video file selected")
