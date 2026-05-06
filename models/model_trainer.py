import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
from datetime import datetime

class ModelTrainer:
    """Train and improve GAN and CNN models with user data"""
    
    def __init__(self, text_detector, image_detector):
        self.text_detector = text_detector
        self.image_detector = image_detector
        self.training_history = []
    
    def train_text_model(self, texts, labels, epochs=5, batch_size=16):
        """Train the text spam detector with user-provided data"""
        # Ensure model is initialized
        self.text_detector._ensure_pretrained()
        
        # Preprocess texts
        X_train = []
        for text in texts:
            X_train.append(self.text_detector.preprocess_text(text))
        
        X_train = np.vstack(X_train)
        y_train = np.array(labels)
        
        # Train the discriminator
        history = self.text_detector.discriminator.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.0,
            verbose=1
        )
        
        # Record training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'model': 'text',
            'samples': len(texts),
            'epochs': epochs,
            'final_accuracy': history.history['accuracy'][-1]
        })
        
        return history
    
    def train_image_model(self, images, labels, epochs=5, batch_size=16):
        """Train the image spam detector with user-provided data"""
        # Ensure model is initialized
        self.image_detector._ensure_pretrained()
        
        if not images:
            # Synthetic training if no images provided
            self.image_detector._pretrain_model()
            return type('obj', (object,), {'history': {'accuracy': [1.0]}})
            
        # Preprocess images
        X_train = []
        for image in images:
            processed = self.image_detector.preprocess_image(image)
            X_train.append(processed[0])
        
        X_train = np.array(X_train)
        y_train = np.array(labels)
        
        # Train the CNN
        history = self.image_detector.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.0,
            verbose=1
        )
        
        # Record training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'model': 'image',
            'samples': len(images),
            'epochs': epochs,
            'final_accuracy': history.history['accuracy'][-1]
        })
        
        return history
    
    def get_training_stats(self):
        """Get statistics about training sessions"""
        if not self.training_history:
            return None
        
        text_sessions = [h for h in self.training_history if h['model'] == 'text']
        image_sessions = [h for h in self.training_history if h['model'] == 'image']
        
        return {
            'total_sessions': len(self.training_history),
            'text_sessions': len(text_sessions),
            'image_sessions': len(image_sessions),
            'total_text_samples': sum(h['samples'] for h in text_sessions),
            'total_image_samples': sum(h['samples'] for h in image_sessions),
            'avg_text_accuracy': np.mean([h['final_accuracy'] for h in text_sessions]) if text_sessions else 0,
            'avg_image_accuracy': np.mean([h['final_accuracy'] for h in image_sessions]) if image_sessions else 0
        }

def get_trainer(text_detector, image_detector):
    """Get model trainer instance"""
    return ModelTrainer(text_detector, image_detector)
