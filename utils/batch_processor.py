import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

class BatchProcessor:
    """Process multiple inputs in batch for efficient spam detection"""
    
    def __init__(self, text_detector, image_detector, ocr_processor):
        self.text_detector = text_detector
        self.image_detector = image_detector
        self.ocr_processor = ocr_processor
    
    def process_text_batch(self, texts, parallel=True):
        """Process multiple text messages in batch"""
        results = []
        
        if parallel and len(texts) > 3:
            # Use thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                predictions = list(executor.map(self.text_detector.predict, texts))
        else:
            predictions = [self.text_detector.predict(text) for text in texts]
        
        for i, (text, prob) in enumerate(zip(texts, predictions)):
            results.append({
                'index': i,
                'text': text[:100],  # First 100 chars
                'spam_probability': prob,
                'is_spam': prob >= 0.5,
                'classification': 'SPAM' if prob >= 0.5 else 'LEGITIMATE'
            })
        
        return results
    
    def process_image_batch(self, images, parallel=True):
        """Process multiple images in batch"""
        results = []
        
        if parallel and len(images) > 3:
            with ThreadPoolExecutor(max_workers=4) as executor:
                predictions = list(executor.map(
                    lambda img: self.image_detector.predict(img),
                    images
                ))
        else:
            predictions = [self.image_detector.predict(img) for img in images]
        
        for i, (image, (prob, features)) in enumerate(zip(images, predictions)):
            results.append({
                'index': i,
                'image_size': f"{image.size[0]}x{image.size[1]}",
                'spam_probability': prob,
                'is_spam': prob >= 0.5,
                'classification': 'SPAM' if prob >= 0.5 else 'LEGITIMATE',
                'edge_density': features.get('edge_density', 0),
                'brightness': features.get('brightness', 0)
            })
        
        return results
    
    def process_mixed_batch(self, items):
        """Process a mixed batch of texts and images"""
        text_results = []
        image_results = []
        
        for item in items:
            if isinstance(item, str):
                prob = self.text_detector.predict(item)
                text_results.append({
                    'type': 'text',
                    'content': item[:100],
                    'spam_probability': prob,
                    'is_spam': prob >= 0.5
                })
            else:
                prob, features = self.image_detector.predict(item)
                image_results.append({
                    'type': 'image',
                    'spam_probability': prob,
                    'is_spam': prob >= 0.5
                })
        
        return {
            'text_results': text_results,
            'image_results': image_results,
            'total_processed': len(items),
            'total_spam': sum(1 for r in text_results + image_results if r['is_spam'])
        }
    
    def create_results_dataframe(self, results):
        """Convert results to pandas DataFrame for easy analysis"""
        return pd.DataFrame(results)

def get_batch_processor(text_detector, image_detector, ocr_processor):
    """Get batch processor instance"""
    return BatchProcessor(text_detector, image_detector, ocr_processor)
