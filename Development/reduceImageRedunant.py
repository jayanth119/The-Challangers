import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import os

class ReduceRedunant:
    # Load ResNet50 as feature extractor
    def load_resnet50(self):
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        return Model(inputs=base_model.input, outputs=base_model.output)

    # Function to extract features from image
    def extract_features(self, img, model):
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img)
        return features.flatten()

    # Function to compute similarity and reduce image list to 5 most similar
    def reduce_images(self, images: np.ndarray, num_images: int = 5):
        if len(images) != 24:
            raise ValueError("Input should contain exactly 24 images.")
        
        # Initialize ResNet50 model
        model = self.load_resnet50()
        
        # Extract features for all images
        image_features = [self.extract_features(img, model) for img in images]
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(image_features)
        
        # Calculate mean similarity for each image (sum of all similarities)
        similarity_scores = np.mean(similarity_matrix, axis=1)
        
        # Print similarity scores for all images
        for idx, score in enumerate(similarity_scores):
            print(f"Similarity score for image {idx + 1}: {score}")
        
        # Select the top 'num_images' with the highest similarity scores (without changing order)
        # We pick the indices of the top `num_images` based on similarity scores.
        most_similar_indices = np.argsort(similarity_scores)[-num_images:]
        
        # Return the reduced list of similar images (without sorting by similarity)
        return images[most_similar_indices]

    # Function to save reduced images to disk
    def save_images(self, images, save_path='reduced_images', num_images=5):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for i, img in enumerate(images):
            # Convert the image (NumPy array) to a PIL Image object
            img_pil = Image.fromarray((img * 255).astype(np.uint8))  # Assuming images are normalized [0, 1]
            img_pil.save(os.path.join(save_path, f'image_{i+1}.png'))
    def generate_test_images(self , num_images=24, image_size=(224, 224, 3)):
        images = []
        for i in range(num_images):
            if i < 10:
                # Create similar images (slight variation)
                base_image = np.random.rand(*image_size) * 0.5 + 0.5  # Base image with mid-level brightness
                # Add small noise to create slight variations
                variation = np.random.rand(*image_size) * 0.1
                images.append(base_image + variation)
            else:
                # Create distinctly different images (e.g., with random noise)
                random_image = np.random.rand(*image_size)
                images.append(random_image)
        
        return np.array(images)

# Example usage
if __name__ == "__main__":
    # Assuming 'images' is a numpy array of shape (24, H, W, C)
    # Replace this with actual image loading process
    
    cl = ReduceRedunant()
    images = cl.generate_test_images(24)

    # Reduce the list of images to 5 most similar (without changing their order)
    reduced_images = cl.reduce_images(images, 5)
    
    # Save the reduced list of images to disk (in their original order)
    cl.save_images(reduced_images, save_path='reduced_images', num_images=5)
    
    print(f"Reduced list of 5 similar images saved to 'reduced_images' folder.")
