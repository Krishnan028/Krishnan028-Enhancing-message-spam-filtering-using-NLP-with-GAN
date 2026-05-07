---
title: SMS Spam Guard AI
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.44.1"
app_file: app.py
pinned: false
python_version: "3.11"
---

# SMS Spam Guard AI

A multi-modal SMS spam detection system built with Streamlit and deep learning. It analyzes **text messages, images, and videos** to identify spam content using a combination of GAN-based text classification, CNN-based image analysis, and OCR text extraction.

## Features

- **Text Spam Detection** – GAN-based discriminator analyzes message text and returns a spam confidence score.
- **Image Spam Detection** – CNN model classifies uploaded images; OCR (Tesseract) extracts any embedded text and re-checks it for spam.
- **Video Spam Detection** – Frames are sampled from videos and each frame is processed through the image pipeline.
- **NLP-based Analysis** – Real-time analysis with progress feedback in the UI.
- **Custom Training** – Drop spam reference images into `training_data/spam_images/` and they will be picked up automatically during model pretraining.
- **Analytics Dashboard** – Tracks detection history, summary statistics, and recent activity.
- **PDF Reporting & Batch Processing** – Generate reports and process multiple inputs at once.
- **Modern UI** – Clean Streamlit interface with status badges, sidebar stats, and sample data demos.

## Tech Stack

- **Frontend / App** – Streamlit
- **Deep Learning** – TensorFlow / Keras (GAN + CNN)
- **Computer Vision** – OpenCV, Pillow
- **OCR** – Pytesseract (Tesseract OCR)
- **Visualization** – Matplotlib, Seaborn
- **Language** – Python 3.11

## Project Structure

```
.
├── app.py                          # Main Streamlit application
├── models/
│   ├── gan_text_detector.py        # GAN-based text spam detector
│   ├── image_spam_detector.py      # CNN-based image spam detector
│   └── model_trainer.py            # Training utilities
├── utils/
│   ├── analytics.py                # Detection history & statistics
│   ├── video_processor.py          # Video frame extraction
│   ├── pdf_reporter.py             # PDF report generation
│   └── batch_processor.py          # Batch processing helpers
├── training_data/
│   └── spam_images/                # Drop real spam images here for training
├── attached_assets/                # Sample / uploaded assets
├── .streamlit/config.toml          # Streamlit configuration
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.11+
- Tesseract OCR installed on the system (required by `pytesseract`)

### Run Locally
```bash
# Install dependencies
pip install streamlit tensorflow opencv-python pytesseract pillow numpy matplotlib seaborn

# Start the app
streamlit run app.py --server.port 5000
```

The app will be available at `http://localhost:5000`.

### Run on Replit
The project is preconfigured. Just click **Run** and Replit will start the Streamlit server on port 5000.

## How It Works

### 1. Text Pipeline
```
Text → Tokenize → Pad → GAN Discriminator → Spam Probability
```

### 2. Image Pipeline
```
Image → Resize (224x224) → CNN → Spam Probability
       └─ OCR → Extract Text → Text Pipeline → Combined Score
```

### 3. Video Pipeline
```
Video → Sample Frames (every 30th, max 10) → Image Pipeline (per frame) → Aggregate
```

## Adding Custom Training Data

To improve spam detection on specific image patterns:

1. Place spam reference images in `training_data/spam_images/`
2. Restart the app
3. The detector will load them, replicate them during pretraining, and learn them strongly as spam

Supported formats: `.jpg`, `.jpeg`, `.png`

## Model Details

### GAN Text Detector
- Vocabulary: 5,000 words
- Sequence length: 100 tokens
- Embedding dimension: 128
- Latent dimension: 100

### CNN Image Detector
- Input: 224 × 224 × 3 RGB images
- Architecture: 3 convolutional blocks (32 → 64 → 128 channels) with batch normalization, max pooling, and dropout
- Dense classification head with sigmoid output

## Use Case

Built as a final-year academic project demonstrating multi-modal spam detection across text, image, and video formats using modern deep learning techniques.

## License

This project is provided for educational purposes.
