import numpy as np
from datetime import datetime
from collections import defaultdict

class AnalyticsTracker:
    """Track and analyze spam detection results"""
    
    def __init__(self):
        self.detection_history = []
        self.daily_stats = defaultdict(lambda: {'spam': 0, 'ham': 0, 'total': 0})
    
    def record_detection(self, input_type, is_spam, confidence, metadata=None):
        """Record a spam detection event"""
        record = {
            'timestamp': datetime.now(),
            'type': input_type,  # 'text', 'image', 'video'
            'is_spam': is_spam,
            'confidence': confidence,
            'metadata': metadata or {}
        }
        
        self.detection_history.append(record)
        
        # Update daily stats
        date_key = record['timestamp'].strftime('%Y-%m-%d')
        self.daily_stats[date_key]['total'] += 1
        if is_spam:
            self.daily_stats[date_key]['spam'] += 1
        else:
            self.daily_stats[date_key]['ham'] += 1
    
    def get_summary_stats(self):
        """Get overall summary statistics"""
        if not self.detection_history:
            return {
                'total_analyzed': 0,
                'total_spam': 0,
                'total_ham': 0,
                'spam_rate': 0,
                'avg_confidence': 0
            }
        
        total = len(self.detection_history)
        spam_count = sum(1 for r in self.detection_history if r['is_spam'])
        confidences = [r['confidence'] for r in self.detection_history]
        
        return {
            'total_analyzed': total,
            'total_spam': spam_count,
            'total_ham': total - spam_count,
            'spam_rate': spam_count / total if total > 0 else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0
        }
    
    def get_type_breakdown(self):
        """Get breakdown by input type"""
        breakdown = defaultdict(lambda: {'total': 0, 'spam': 0, 'ham': 0})
        
        for record in self.detection_history:
            input_type = record['type']
            breakdown[input_type]['total'] += 1
            if record['is_spam']:
                breakdown[input_type]['spam'] += 1
            else:
                breakdown[input_type]['ham'] += 1
        
        return dict(breakdown)
    
    def get_daily_trends(self, days=7):
        """Get daily detection trends"""
        sorted_dates = sorted(self.daily_stats.keys())[-days:]
        
        trends = []
        for date in sorted_dates:
            stats = self.daily_stats[date]
            trends.append({
                'date': date,
                'total': stats['total'],
                'spam': stats['spam'],
                'ham': stats['ham'],
                'spam_rate': stats['spam'] / stats['total'] if stats['total'] > 0 else 0
            })
        
        return trends
    
    def get_confidence_distribution(self, bins=10):
        """Get distribution of confidence scores"""
        if not self.detection_history:
            return {'bins': [], 'counts': []}
        
        confidences = [r['confidence'] for r in self.detection_history]
        counts, bin_edges = np.histogram(confidences, bins=bins, range=(0, 1))
        
        return {
            'bins': bin_edges[:-1].tolist(),
            'counts': counts.tolist()
        }
    
    def get_stats(self):
        """Standardized statistics for the dashboard"""
        summary = self.get_summary_stats()
        return {
            'total': summary['total_analyzed'],
            'spam': summary['total_spam'],
            'ham': summary['total_ham'],
            'spam_rate': summary['spam_rate'] * 100,
            'avg_confidence': summary['avg_confidence']
        }
    
    def log_detection(self, input_type, source, is_spam, confidence):
        """Compatibility wrapper for record_detection"""
        self.record_detection(input_type, is_spam, confidence, {'source': source})

    def get_history_df(self):
        """Get history as a pandas DataFrame for Streamlit display"""
        import pandas as pd
        if not self.detection_history:
            return pd.DataFrame()
        
        data = []
        for r in self.detection_history:
            data.append({
                'Timestamp': r['timestamp'],
                'Type': r['type'],
                'Is Spam': r['is_spam'],
                'Confidence': r['confidence'],
                'Source': r['metadata'].get('source', 'Unknown')
            })
        return pd.DataFrame(data)

    def generate_summary_plot(self):
        """Generate a summary plot using matplotlib/seaborn"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not self.detection_history:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data yet", ha='center', va='center')
            return fig
            
        df = self.get_history_df()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x='Type', hue='Is Spam', ax=ax)
        ax.set_title('Detections by Media Type')
        return fig

    def clear_history(self):
        """Clear all analytics history"""
        self.detection_history = []
        self.daily_stats = defaultdict(lambda: {'spam': 0, 'ham': 0, 'total': 0})

def get_analytics_tracker():
    """Get analytics tracker instance"""
    return AnalyticsTracker()
