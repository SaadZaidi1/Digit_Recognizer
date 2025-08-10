import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

class ImprovedDigitRecognizer:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
    def load_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Normalize pixel values
        self.x_train = x_train.astype('float32') / 255.0
        self.x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        self.x_train = self.x_train.reshape(-1, 28, 28, 1)
        self.x_test = self.x_test.reshape(-1, 28, 28, 1)
        
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set shape: {self.x_train.shape}")
        print(f"Test set shape: {self.x_test.shape}")
        
    def create_model(self):
        """Create an improved CNN model"""
        model = tf.keras.Sequential([
            # First Convolutional Block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second Convolutional Block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third Convolutional Block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile model with better optimizer and learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model architecture:")
        model.summary()
        
    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])
        return data_augmentation
        
    def train_model(self, epochs=15, batch_size=32):
        """Train the model with data augmentation and callbacks"""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
            
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with data augmentation
        data_augmentation = self.create_data_augmentation()
        
        self.history = self.model.fit(
            data_augmentation(self.x_train),
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
    def evaluate_model(self):
        """Evaluate the model and print detailed metrics"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        # Load best model
        if os.path.exists('best_model.h5'):
            self.model = tf.keras.models.load_model('best_model.h5')
            print("Loaded best model from training.")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Detailed metrics
        accuracy = accuracy_score(self.y_test, y_pred_classes)
        print(f"\nAccuracy Score: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_classes))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_accuracy, accuracy
        
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, filename='improved_digit_model.h5'):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filename)
            print(f"Model saved as {filename}")
        else:
            print("No model to save.")
            
    def load_model(self, filename='improved_digit_model.h5'):
        """Load a trained model"""
        if os.path.exists(filename):
            self.model = tf.keras.models.load_model(filename)
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file {filename} not found.")
            
    def predict_digit(self, image):
        """Predict a single digit image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to grayscale
            image = np.mean(image, axis=2)
            image = image.reshape(1, 28, 28, 1)
            
        # Normalize
        image = image.astype('float32') / 255.0
        
        # Predict
        prediction = self.model.predict(image, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return predicted_digit, confidence, prediction[0]

def main():
    """Main function to train and evaluate the improved model"""
    # Initialize the recognizer
    recognizer = ImprovedDigitRecognizer()
    
    # Load data
    recognizer.load_data()
    
    # Create and train model
    recognizer.create_model()
    recognizer.train_model(epochs=15)
    
    # Evaluate model
    test_acc, acc_score = recognizer.evaluate_model()
    
    # Plot training history
    recognizer.plot_training_history()
    
    # Save the model
    recognizer.save_model()
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print("Model training and evaluation completed!")

if __name__ == "__main__":
    main()
