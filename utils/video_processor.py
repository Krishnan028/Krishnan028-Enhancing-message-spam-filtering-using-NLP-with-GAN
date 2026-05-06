import cv2
import numpy as np
from PIL import Image
import tempfile
import os

class VideoProcessor:
    """Process video files for spam detection"""
    
    def __init__(self):
        self.frame_interval = 30  # Extract every 30th frame
        self.max_frames = 10  # Maximum frames to analyze
    
    def extract_frames(self, video_path, num_frames=None):
        """Extract frames from video"""
        if num_frames is None:
            num_frames = self.max_frames
            
        frames = []
        frame_numbers = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return frames, frame_numbers
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame interval to get evenly distributed frames
            if total_frames > num_frames:
                interval = total_frames // num_frames
            else:
                interval = 1
            
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < num_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                    frame_numbers.append(frame_count)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            return frames, frame_numbers
            
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            return frames, frame_numbers
    
    def get_video_info(self, video_path):
        """Get video metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            info = {
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0
            }
            
            if info['fps'] > 0:
                info['duration'] = info['total_frames'] / info['fps']
            
            cap.release()
            return info
            
        except Exception as e:
            print(f"Error getting video info: {str(e)}")
            return None
    
    def analyze_video_frames(self, frames, image_detector, text_detector, ocr_processor):
        """Analyze extracted frames for spam content"""
        results = {
            'frame_count': len(frames),
            'spam_frames': 0,
            'avg_image_spam_score': 0,
            'avg_text_spam_score': 0,
            'text_detected': False,
            'extracted_texts': [],
            'frame_scores': []
        }
        
        if not frames:
            return results
        
        image_scores = []
        text_scores = []
        
        for i, frame in enumerate(frames):
            # Image-based spam detection
            img_spam_score, _ = image_detector.predict(frame)
            image_scores.append(img_spam_score)
            
            # OCR and text analysis
            ocr_result = ocr_processor.extract_text(frame)
            
            if ocr_result['text']:
                results['text_detected'] = True
                results['extracted_texts'].append({
                    'frame': i,
                    'text': ocr_result['text'][:100],  # First 100 chars
                    'confidence': ocr_result['confidence']
                })
                
                # Analyze extracted text
                text_spam_score = text_detector.predict(ocr_result['text'])
                text_scores.append(text_spam_score)
            
            # Track frame scores
            results['frame_scores'].append({
                'frame': i,
                'image_score': float(img_spam_score),
                'text_score': float(text_scores[-1]) if text_scores and len(text_scores) > i else 0
            })
            
            if img_spam_score > 0.5:
                results['spam_frames'] += 1
        
        results['avg_image_spam_score'] = float(np.mean(image_scores)) if image_scores else 0
        results['avg_text_spam_score'] = float(np.mean(text_scores)) if text_scores else 0
        
        return results

def get_video_processor():
    """Get video processor instance"""
    return VideoProcessor()
