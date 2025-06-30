import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def load_images_from_dir(directory, target_size=(500, 500), crop=False):
    images = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

            if crop:
                # Crop the image to the specified aspect ratio
                h, w, _ = img.shape
                target_h, target_w = target_size

                # Calculate aspect ratios
                aspect_ratio_img = w / h
                aspect_ratio_target = target_w / target_h

                if aspect_ratio_img > aspect_ratio_target:
                    # Crop horizontally
                    new_w = int(h * aspect_ratio_target)
                    left = (w - new_w) // 2
                    img = img[:, left:left + new_w, :]
                elif aspect_ratio_img < aspect_ratio_target:
                    # Crop vertically
                    new_h = int(w / aspect_ratio_target)
                    top = (h - new_h) // 2
                    img = img[top:top + new_h, :, :]

            else:
                # Resize while preserving aspect ratio and pad to target size
                img = cv2.resize(img, target_size)
                h, w, _ = img.shape
                pad_top = (target_size[0] - h) // 2
                pad_bottom = target_size[0] - h - pad_top
                pad_left = (target_size[1] - w) // 2
                pad_right = target_size[1] - w - pad_left
                img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

            images.append(img)
    return np.array(images)



# Function to build the deblurring model
def build_deblur_model(input_shape):
    model = Sequential()

    # Encoder
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Decoder
    model.add(Conv2DTranspose(3, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('linear'))

    return model

# Function to train the deblurring model
def train_deblur_model(train_blurred_images, train_sharp_images, epochs, batch_size):
    input_shape = train_blurred_images[0].shape

    # Build the deblurring model
    model = build_deblur_model(input_shape)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model without data augmentation for simplicity
    history = model.fit(
        train_blurred_images / 255.0,
        train_sharp_images / 255.0,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )

    return model

# Function to preprocess the image for deblurring
def preprocess_image(image_path, target_size=(500, 500)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Calculate aspect ratios
    h, w, _ = img.shape
    target_h, target_w = target_size
    aspect_ratio_img = w / h
    aspect_ratio_target = target_w / target_h

    if aspect_ratio_img > aspect_ratio_target:
        # Resize based on width
        new_w = target_w
        new_h = int(new_w / aspect_ratio_img)
    else:
        # Resize based on height
        new_h = target_h
        new_w = int(new_h * aspect_ratio_img)

    img = cv2.resize(img, (new_w, new_h))  # Resize to maintain aspect ratio
    
    # Pad the image to match the model input shape
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    
    return img / 255.0  # Normalize to [0, 1]

# Directory paths for blurred and sharp images
blurred_images_dir = './blurry'
sharp_images_dir = './unblurry'

# Load blurred image
blurred_img = preprocess_image('./Blurry_Image.jpg')

# Check if the trained model file exists
model_filename = 'deblur_model.h5'
if os.path.exists(model_filename):
    # Load the trained model
    model = tf.keras.models.load_model(model_filename)
else:
    # Load training data (blurred images and corresponding sharp images)
    train_blurred_images = load_images_from_dir(blurred_images_dir)
    train_sharp_images = load_images_from_dir(sharp_images_dir)

    # Train the model
    model = train_deblur_model(train_blurred_images, train_sharp_images, epochs=20, batch_size=32)
    # Save the trained model
    model.save(model_filename)

# Save the trained model
model.save('deblur_model.h5')

# Load the trained model
# model = tf.keras.models.load_model('deblur_model.h5')

# Deblur the image using the trained model
deblurred_img = model.predict(np.expand_dims(blurred_img, axis=0))

# Post-processing (Optional)
deblurred_img = np.squeeze(deblurred_img, axis=0)
deblurred_img = np.clip(deblurred_img * 255, 0, 255).astype(np.uint8)  # Scale back to 0-255 range

# Resize deblurred image to match the original size
deblurred_img = cv2.resize(deblurred_img, (blurred_img.shape[1], blurred_img.shape[0]))

# Display the deblurred image
plt.imshow(deblurred_img)
plt.axis('off')  # Turn off axis labels
plt.show()

# Save the deblurred image
Image.fromarray(deblurred_img).save('./Deblurred_Image.jpg')
cv2.waitKey(0)
cv2.destroyAllWindows()

