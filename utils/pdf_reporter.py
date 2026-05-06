from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import io

class PDFReporter:
    """Generate PDF reports for spam detection results"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12
        ))
    
    def generate_detection_report(self, results, input_type, summary_stats=None):
        """Generate a PDF report for detection results"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        story = []
        
        # Title
        title = Paragraph("SMS Spam Detection Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Metadata
        date_str = datetime.now().strftime("%B %d, %Y at %H:%M")
        metadata = Paragraph(f"<b>Generated:</b> {date_str}<br/>"
                           f"<b>Analysis Type:</b> {input_type.upper()}<br/>"
                           f"<b>Total Samples:</b> {len(results)}",
                           self.styles['Normal'])
        story.append(metadata)
        story.append(Spacer(1, 20))
        
        # Summary Statistics
        if summary_stats:
            story.append(Paragraph("Summary Statistics", self.styles['SectionHeader']))
            summary_data = [
                ['Metric', 'Value'],
                ['Total Analyzed', str(summary_stats.get('total_analyzed', len(results)))],
                ['Spam Detected', str(summary_stats.get('total_spam', 0))],
                ['Legitimate Messages', str(summary_stats.get('total_ham', 0))],
                ['Spam Rate', f"{summary_stats.get('spam_rate', 0):.1%}"],
                ['Average Confidence', f"{summary_stats.get('avg_confidence', 0):.1%}"]
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))
        
        # Detection Results
        story.append(Paragraph("Detection Results", self.styles['SectionHeader']))
        story.append(Spacer(1, 12))
        
        # Create results table
        if input_type == 'text':
            results_data = [['#', 'Text Preview', 'Classification', 'Confidence']]
            for r in results[:50]:  # Limit to 50 for PDF size
                results_data.append([
                    str(r.get('index', 0) + 1),
                    r.get('text', '')[:60] + '...' if len(r.get('text', '')) > 60 else r.get('text', ''),
                    r.get('classification', 'N/A'),
                    f"{r.get('spam_probability', 0):.1%}"
                ])
            
            col_widths = [0.5*inch, 3.5*inch, 1*inch, 1*inch]
        else:
            results_data = [['#', 'Image Size', 'Classification', 'Confidence', 'Edge Density']]
            for r in results[:50]:
                results_data.append([
                    str(r.get('index', 0) + 1),
                    r.get('image_size', 'N/A'),
                    r.get('classification', 'N/A'),
                    f"{r.get('spam_probability', 0):.1%}",
                    f"{r.get('edge_density', 0):.3f}"
                ])
            
            col_widths = [0.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1*inch]
        
        results_table = Table(results_data, colWidths=col_widths)
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        # Color code spam vs legitimate
        for i, r in enumerate(results[:50], start=1):
            if r.get('is_spam'):
                results_table.setStyle(TableStyle([
                    ('BACKGROUND', (2, i), (2, i), colors.HexColor('#ffe6e6'))
                ]))
        
        story.append(results_table)
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = Paragraph(
            "<i>This report was generated by the SMS Spam Detection System using "
            "GAN-based text classification and CNN-based image analysis.</i>",
            self.styles['Normal']
        )
        story.append(footer_text)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_analytics_report(self, analytics_data, training_stats=None):
        """Generate a comprehensive analytics report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        story = []
        
        # Title
        title = Paragraph("Spam Detection Analytics Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Date
        date_str = datetime.now().strftime("%B %d, %Y at %H:%M")
        metadata = Paragraph(f"<b>Generated:</b> {date_str}", self.styles['Normal'])
        story.append(metadata)
        story.append(Spacer(1, 20))
        
        # Overall Statistics
        story.append(Paragraph("Overall Statistics", self.styles['SectionHeader']))
        summary_data = [
            ['Metric', 'Value'],
            ['Total Analyzed', str(analytics_data.get('total_analyzed', 0))],
            ['Spam Detected', str(analytics_data.get('total_spam', 0))],
            ['Legitimate Messages', str(analytics_data.get('total_ham', 0))],
            ['Spam Rate', f"{analytics_data.get('spam_rate', 0):.1%}"],
            ['Average Confidence', f"{analytics_data.get('avg_confidence', 0):.1%}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Training Statistics (if available)
        if training_stats:
            story.append(PageBreak())
            story.append(Paragraph("Model Training Statistics", self.styles['SectionHeader']))
            
            training_data = [
                ['Metric', 'Value'],
                ['Total Training Sessions', str(training_stats.get('total_sessions', 0))],
                ['Text Model Sessions', str(training_stats.get('text_sessions', 0))],
                ['Image Model Sessions', str(training_stats.get('image_sessions', 0))],
                ['Total Text Samples', str(training_stats.get('total_text_samples', 0))],
                ['Total Image Samples', str(training_stats.get('total_image_samples', 0))],
                ['Avg Text Accuracy', f"{training_stats.get('avg_text_accuracy', 0):.1%}"],
                ['Avg Image Accuracy', f"{training_stats.get('avg_image_accuracy', 0):.1%}"]
            ]
            
            training_table = Table(training_data, colWidths=[3*inch, 2*inch])
            training_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(training_table)
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = Paragraph(
            "<i>SMS Spam Detection System - Final Year Project<br/>"
            "Powered by GAN and CNN Deep Learning Models</i>",
            self.styles['Normal']
        )
        story.append(footer_text)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer

def get_pdf_reporter():
    """Get PDF reporter instance"""
    return PDFReporter()
