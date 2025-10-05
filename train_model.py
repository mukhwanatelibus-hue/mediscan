import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import json
from sklearn.model_selection import train_test_split
import random
import shutil
from urllib.request import urlretrieve
import zipfile

class PillModelTrainer:
    def __init__(self):
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 30
        self.class_names = []  # Will be auto-detected
        
    def detect_classes(self, data_dir):
        """Detect available classes from folder structure"""
        classes = []
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                if os.path.isdir(os.path.join(data_dir, item)):
                    classes.append(item)
        return sorted(classes)
    
    def download_negative_examples(self, data_dir):
        """Download negative examples (non-pill images) if needed"""
        negative_dir = os.path.join(data_dir, "not_a_pill")
        os.makedirs(negative_dir, exist_ok=True)
        
        # Check if we already have negative examples
        existing_files = [f for f in os.listdir(negative_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(existing_files) >= 100:
            print(f"Already have {len(existing_files)} negative examples")
            return
            
        print("Downloading negative examples...")
        
        # Sample URLs for diverse negative examples (people, objects, backgrounds)
        sample_urls = [
            "https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg",
            "https://storage.googleapis.com/download.tensorflow.org/example_images/194px-Android_robot.png",
            "https://storage.googleapis.com/download.tensorflow.org/example_images/192px-People's_Flag_of_Minnesota.svg.png"
        ]
        
        # Try to download some sample negative images
        for i, url in enumerate(sample_urls):
            try:
                filename = os.path.join(negative_dir, f"negative_sample_{i}.jpg")
                urlretrieve(url, filename)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Could not download {url}: {e}")
        
        # If we still don't have enough negative examples, create some programmatically
        if len([f for f in os.listdir(negative_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) < 50:
            self.generate_synthetic_negatives(negative_dir)
    
    def generate_synthetic_negatives(self, negative_dir):
        """Generate synthetic negative examples"""
        print("Generating synthetic negative examples...")
        
        # Create various types of non-pill images
        for i in range(50):
            # Random background color
            bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = np.full((224, 224, 3), bg_color, dtype=np.uint8)
            
            # Add some random shapes/textures
            if random.random() > 0.5:
                # Add random rectangles
                for _ in range(random.randint(1, 5)):
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    pt1 = (random.randint(0, 200), random.randint(0, 200))
                    pt2 = (random.randint(10, 224), random.randint(10, 224))
                    cv2.rectangle(img, pt1, pt2, color, -1)
            
            if random.random() > 0.5:
                # Add random circles
                for _ in range(random.randint(1, 3)):
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    center = (random.randint(20, 204), random.randint(20, 204))
                    radius = random.randint(5, 50)
                    cv2.circle(img, center, radius, color, -1)
            
            # Add some text
            if random.random() > 0.7:
                font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX])
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.putText(img, "Text", (50, 150), font, 1, color, 2)
            
            # Save the synthetic negative example
            cv2.imwrite(os.path.join(negative_dir, f"synthetic_negative_{i}.jpg"), img)
        
        print(f"Generated 50 synthetic negative examples")
        
    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess pill images"""
        images = []
        labels = []
        
        self.class_names = self.detect_classes(data_dir)
        if not self.class_names:
            raise ValueError(f"No class folders found in {data_dir}")
            
        print(f"Detected classes: {self.class_names}")
        
        # Ensure we have a balanced dataset
        class_counts = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            
            image_count = 0
            valid_images = []
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Basic check if image is valid (not corrupted)
                        if img.size > 0 and len(img.shape) == 3:
                            valid_images.append(img)
                            image_count += 1
            
            # Limit the number of samples per class to avoid imbalance
            max_samples = 500  # Maximum samples per class
            if len(valid_images) > max_samples:
                valid_images = random.sample(valid_images, max_samples)
                image_count = max_samples
                
            for img in valid_images:
                img = cv2.resize(img, self.img_size)
                img = img / 255.0  # Normalize
                images.append(img)
                labels.append(class_idx)
            
            class_counts[class_name] = image_count
            print(f"Loaded {image_count} images from {class_name}")
        
        if len(images) == 0:
            raise ValueError("No images found in data directory")
            
        # Print class distribution
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")
            
        return np.array(images), np.array(labels)
    
    def augment_data(self, images, labels):
        """Enhanced data augmentation"""
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(images, labels):
            # Original image
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Flip horizontally
            flipped = cv2.flip(img, 1)
            augmented_images.append(flipped)
            augmented_labels.append(label)
            
            # Flip vertically
            flipped_vert = cv2.flip(img, 0)
            augmented_images.append(flipped_vert)
            augmented_labels.append(label)
            
            # Adjust brightness
            bright = np.clip(img * (0.8 + random.random() * 0.4), 0, 1)
            augmented_images.append(bright)
            augmented_labels.append(label)
            
            # Rotate slightly
            angle = random.randint(-15, 15)
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            augmented_images.append(rotated)
            augmented_labels.append(label)
            
            # Add slight blur
            if random.random() > 0.7:
                ksize = random.choice([3, 5])
                blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
                augmented_images.append(blurred)
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def create_model(self, num_classes):
        """Create improved CNN model for pill classification"""
        # Use a pre-trained model as base for better feature extraction
        base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self):
        """Train the pill classification model"""
        print("Starting model training...")
        
        # Check if data directory exists
        data_dir = 'data/train'
        if not os.path.exists(data_dir):
            print(f"❌ Error: Directory {data_dir} does not exist!")
            print("Creating directory structure...")
            os.makedirs(data_dir, exist_ok=True)
            print("Please add your medicine images to data/train/medicine_name/")
            return None, None
        
        # Download/generate negative examples
        self.download_negative_examples(data_dir)
        
        print("Loading training data...")
        try:
            X, y = self.load_and_preprocess_data(data_dir)
            print(f"Successfully loaded {len(X)} images")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
        
        # Data augmentation
        X_augmented, y_augmented = self.augment_data(X, y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create and train model
        print("Creating model...")
        model = self.create_model(len(self.class_names))
        
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
                keras.callbacks.ModelCheckpoint(
                    'models/best_model.h5', 
                    save_best_only=True, 
                    monitor='val_accuracy'
                )
            ],
            verbose=1
        )
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        print("Saving model...")
        model.save('models/pill_classifier.h5')
        
        # Save model info
        model_info = {
            "class_names": self.class_names,
            "input_size": self.img_size,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "validation_accuracy": float(history.history['val_accuracy'][-1]),
            "training_accuracy": float(history.history['accuracy'][-1]),
            "notes": "Model includes negative examples (not_a_pill class) to reduce false positives"
        }
        
        with open('models/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✅ Training completed successfully!")
        print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        # Fine-tuning: Unfreeze some layers for additional training
        print("Starting fine-tuning...")
        model.trainable = True
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        fine_tune_history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=10,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Save fine-tuned model
        model.save('models/pill_classifier_finetuned.h5')
        print(f"Fine-tuned Validation Accuracy: {fine_tune_history.history['val_accuracy'][-1]:.4f}")
        
        return model, history

# Also add this function to create a basic dataset structure
def create_dataset_structure():
    """Create the basic folder structure for training data"""
    base_dir = 'data/train'
    classes = ['amoxicillin', 'ibuprofen', 'paracetamol', 'not_a_pill']
    
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Created directory: {class_dir}")
    
    print("\nPlease add your images to these folders:")
    print("- amoxicillin/: Images of amoxicillin pills")
    print("- ibuprofen/: Images of ibuprofen pills") 
    print("- paracetamol/: Images of paracetamol pills")
    print("- not_a_pill/: Images of things that are NOT pills (people, objects, etc.)")
    print("\nThen run the training script again.")

if __name__ == "__main__":
    print("=" * 60)
    print("Medicine Pill Classification Trainer")
    print("Now with negative examples to prevent false positives!")
    print("=" * 60)
    
    # Create dataset structure if it doesn't exist
    if not os.path.exists('data/train'):
        create_dataset_structure()
    else:
        trainer = PillModelTrainer()
        model, history = trainer.train()
        
        if model is not None:
            print("✅ Model trained and saved successfully!")
            print("Location: models/pill_classifier.h5")
            print("Fine-tuned model: models/pill_classifier_finetuned.h5")
            print("\nYou can now run the app with: python main.py")
        else:
            print("❌ Model training failed!")
            print("Please check your data folder structure.")