import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import cv2

class ImageSpamDetector:
    """CNN-based Image Spam Detector"""
    
    def __init__(self):
        self.img_height = 224
        self.img_width = 224
        self.model = None
        self._initialize_model()
        
    def build_cnn_model(self):
        """Build CNN model for image spam detection"""
        model = keras.Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 3)),
            
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ], name='image_spam_detector')
        
        return model
    
    def _initialize_model(self):
        """Initialize and compile the CNN model"""
        self.model = self.build_cnn_model()
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self._is_pretrained = False
    
    def _ensure_pretrained(self):
        """Ensure the model is pretrained before first use"""
        if not self._is_pretrained:
            self._pretrain_model()
            self._is_pretrained = True
    
    def _generate_synthetic_samples(self, is_spam, num_samples=50):
        """Generate synthetic images with characteristics of spam/ham"""
        samples = []
        labels = []
        
        for _ in range(num_samples):
            if is_spam:
                # Spam images: high saturation, many edges, bright colors
                # Create a colorful, text-heavy synthetic image
                img = np.random.randint(100, 255, (self.img_height, self.img_width, 3), dtype=np.uint8)
                
                # Add high frequency patterns (simulating text/edges)
                for i in range(10):
                    x, y = np.random.randint(0, self.img_width-50), np.random.randint(0, self.img_height-50)
                    img[y:y+5, x:x+50] = 255
                    
                # Increase saturation
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
            else:
                # Ham images: normal photos, less edges, moderate colors
                # Create natural-looking synthetic image
                img = np.random.randint(50, 200, (self.img_height, self.img_width, 3), dtype=np.uint8)
                
                # Add smooth gradients (simulating natural photos)
                gradient = np.linspace(0, 1, self.img_width)
                for c in range(3):
                    img[:, :, c] = np.clip(img[:, :, c] * gradient, 0, 255).astype(np.uint8)
            
            # Normalize
            img_normalized = img.astype(np.float32) / 255.0
            samples.append(img_normalized)
            labels.append(1 if is_spam else 0)
        
        return np.array(samples), np.array(labels)
    
    def _load_real_spam_samples(self):
        """Load real spam image samples from training_data/spam_images/"""
        samples = []
        spam_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training_data', 'spam_images')
        if not os.path.isdir(spam_dir):
            return np.array([])
        patterns = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(spam_dir, p)))
        for fp in files:
            try:
                img = Image.open(fp).convert('RGB').resize((self.img_width, self.img_height))
                samples.append(np.array(img).astype(np.float32) / 255.0)
            except Exception:
                continue
        return np.array(samples) if samples else np.array([])

    def _pretrain_model(self):
        """Pretrain the CNN with synthetic spam/ham image patterns"""
        # Generate synthetic training data
        spam_images, spam_labels = self._generate_synthetic_samples(is_spam=True, num_samples=100)
        ham_images, ham_labels = self._generate_synthetic_samples(is_spam=False, num_samples=100)
        
        # Load real spam samples and replicate them so the model learns them strongly
        real_spam = self._load_real_spam_samples()
        if len(real_spam) > 0:
            replicated = np.repeat(real_spam, 30, axis=0)
            real_labels = np.ones(len(replicated), dtype=np.int32)
            X_train = np.vstack([spam_images, ham_images, replicated])
            y_train = np.concatenate([spam_labels, ham_labels, real_labels])
        else:
            X_train = np.vstack([spam_images, ham_images])
            y_train = np.concatenate([spam_labels, ham_labels])
        
        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Train the model using manual batches to avoid TF iterator bugs
        epochs = 5
        batch_size = 32
        n_samples = len(X_train)
        for _ in range(epochs):
            perm = np.random.permutation(n_samples)
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                self.model.train_on_batch(X_shuf[start:end], y_shuf[start:end])
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize image
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
            
        img = img.convert('RGB')
        img = img.resize((self.img_width, self.img_height))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        return np.expand_dims(img_array, axis=0)
    
    def analyze_image_features(self, image):
        """Analyze image features for spam indicators including screenshot detection"""
        img_array = np.array(image)
        
        # Convert to different color spaces
        if len(img_array.shape) == 2:
            gray = img_array
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        features = {}
        
        # Color analysis
        if len(img_array.shape) == 3:
            # Brightness
            brightness = np.mean(img_array)
            features['brightness'] = brightness
            
            # Color saturation
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1])
            features['saturation'] = saturation
            
            # Color variance
            color_std = np.std(img_array)
            features['color_variance'] = color_std
        
        # Edge detection (spam images often have many edges/text)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features['edge_density'] = edge_density
        
        # Contrast
        contrast = np.std(gray)
        features['contrast'] = contrast
        
        # -------------------------------------------------------------------
        # Advanced: Text region detection (screenshots with text)
        # -------------------------------------------------------------------
        # Detect horizontal lines of text-like content using morphology
        try:
            h, w = gray.shape[:2]
            
            # Binary threshold for text detection
            _, text_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Horizontal morphological operation to find text lines
            horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 8, 20), 1))
            horiz_lines = cv2.morphologyEx(text_binary, cv2.MORPH_OPEN, horiz_kernel)
            text_line_density = np.sum(horiz_lines > 0) / horiz_lines.size
            features['text_line_density'] = text_line_density
            
            # Count distinct text-like contours (small blobs = characters)
            contours, _ = cv2.findContours(text_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            char_like = 0
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(ch, 1)
                area = cw * ch
                # Character-like: small, reasonable aspect ratio
                if 4 < area < (h * w * 0.05) and 0.1 < aspect < 10:
                    char_like += 1
            features['text_contour_count'] = char_like
            features['text_contour_density'] = char_like / max(w * h / 1000, 1)
            
            # Screenshot detection: check for UI-like patterns
            # - Status bar region (top ~8% of image)
            # - Chat bubble patterns (rounded rectangles)
            # - Uniform colored header/footer bands
            top_strip = gray[:max(h // 12, 10), :]
            bottom_strip = gray[h - max(h // 12, 10):, :]
            top_uniformity = 1.0 - (np.std(top_strip) / 128.0)
            bottom_uniformity = 1.0 - (np.std(bottom_strip) / 128.0)
            features['top_uniformity'] = max(top_uniformity, 0)
            features['bottom_uniformity'] = max(bottom_uniformity, 0)
            
            # Detect if image looks like a phone screenshot
            # (uniform header bar + high text density below)
            is_screenshot_like = (top_uniformity > 0.6 and char_like > 30)
            features['screenshot_like'] = 1.0 if is_screenshot_like else 0.0
            
        except Exception:
            features['text_line_density'] = 0
            features['text_contour_count'] = 0
            features['text_contour_density'] = 0
            features['top_uniformity'] = 0
            features['bottom_uniformity'] = 0
            features['screenshot_like'] = 0
        
        return features
    
    def predict(self, image):
        """Predict if image contains spam — feature-based engine + CNN ensemble"""
        self._ensure_pretrained()

        processed_img = self.preprocess_image(image)
        cnn_prediction = float(self.model.predict(processed_img, verbose=0)[0][0])

        features = self.analyze_image_features(image)
        feature_score = 0.0

        # Edge density: text-heavy images (spam flyers, screenshots) have many edges
        edge = features.get('edge_density', 0)
        if edge > 0.25:
            feature_score += 0.30
        elif edge > 0.15:
            feature_score += 0.18
        elif edge > 0.08:
            feature_score += 0.08

        # Very high saturation → promotional / advertising image
        saturation = features.get('saturation', 0)
        if saturation > 160:
            feature_score += 0.18
        elif saturation > 120:
            feature_score += 0.08

        # Very bright (white-background spam flyers)
        brightness = features.get('brightness', 128)
        if brightness > 210:
            feature_score += 0.12

        # High color variance → busy promotional design
        if features.get('color_variance', 0) > 80:
            feature_score += 0.10

        # High contrast → attention-grabbing design
        if features.get('contrast', 0) > 80:
            feature_score += 0.10

        # -------------------------------------------------------------------
        # NEW: Screenshot & text-heavy image boosters
        # -------------------------------------------------------------------
        # Lots of text-like contours → likely screenshot with text
        text_contours = features.get('text_contour_count', 0)
        if text_contours > 100:
            feature_score += 0.25
        elif text_contours > 50:
            feature_score += 0.15
        elif text_contours > 20:
            feature_score += 0.08
        
        # Text line density (horizontal runs of text)
        text_line_density = features.get('text_line_density', 0)
        if text_line_density > 0.05:
            feature_score += 0.15
        elif text_line_density > 0.02:
            feature_score += 0.08
        
        # Looks like a phone screenshot (uniform header + text)
        if features.get('screenshot_like', 0) > 0.5:
            feature_score += 0.20

        # Ensemble: feature-based (65%) + CNN (35%)
        combined_probability = 0.35 * cnn_prediction + 0.65 * min(feature_score, 1.0)
        combined_probability = float(max(0.02, min(0.98, combined_probability)))

        return combined_probability, features
    
    def get_model_info(self):
        """Get information about the model architecture"""
        return {
            'type': 'Convolutional Neural Network',
            'architecture': '3 Conv Blocks + Dense',
            'input_shape': (self.img_height, self.img_width, 3),
            'trainable_params': self.model.count_params(),
            'layers': len(self.model.layers)
        }

def get_model():
    """Get or initialize the image spam detector"""
    detector = ImageSpamDetector()
    return detector
