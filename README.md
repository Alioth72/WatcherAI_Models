# AI-Powered Video Surveillance Suite

This project is a comprehensive suite of tools for advanced video analysis, combining real-time anomaly detection, person re-identification, and natural language video search capabilities. It leverages state-of-the-art AI models to transform raw video footage into structured, searchable, and actionable intelligence.

## Key Features

  * **Live Anomaly Detection**: Monitors video streams (RTSP or local files) in real-time to detect suspicious activities.
      * **Two-Stage Detection**: Uses a lightweight ResNet+SVM model for initial, rapid detection, and escalates to a powerful Vision-Language Model (Google's Gemini) for detailed forensic analysis.
      * **Automated Alerts**: Automatically sends detailed WhatsApp alerts with video clips of the anomaly upon confirmation by the VLM.
  * **Person Re-Identification (ReID)**: Detects and assigns unique, persistent IDs to individuals in a video feed, enabling tracking across different frames and times.
  * **Natural Language Video Search**: An intelligent retrieval system that allows you to search through your entire video library using plain English queries.
      * **AI-Powered Indexing**: Automatically processes videos, generating rich, descriptive summaries for every scene using Gemini.
      * **Vector Search**: Embeds scene descriptions and user queries into a vector space using FAISS for lightning-fast semantic search.
      * **Interactive UI**: A Streamlit-based web interface to perform searches and view the retrieved video clips and keyframes.

## System Architecture

The project is divided into two main components: `LiveAnomalyDetection` and `VideoDescripter+QueryRetrieval`.

### 1\. Live Anomaly Detection Workflow

This pipeline is designed for real-time monitoring and alerting.

`Video Input (RTSP/File) -> Frame Sampling & Batching -> [ResNet50 + SVM] Fast Anomaly Check -> If Anomaly Persists -> [Google Gemini] VLM Forensic Analysis -> [Meta API] WhatsApp Alert`

### 2\. Video Search & Retrieval Workflow

This system works in two phases: offline indexing and online retrieval.

**Offline Indexing:**
`Video Library -> Frame Batching -> [Google Gemini] Scene Description -> [SentenceTransformer] Text Embedding -> [FAISS] Save Vector Index`

**Online Retrieval:**
`User Query -> [SentenceTransformer] Embed Query -> [FAISS] Vector Search -> Retrieve & Display Matched Video Clips`

-----

##  Technologies Used

  * **AI / ML**: PyTorch, TensorFlow, Google Gemini, `ultralytics` (YOLOv8), `sentence-transformers`, `faiss-cpu`, Scikit-learn
  * **Video Processing**: OpenCV, `vidgear`, `ffmpeg`
  * **Web & API**: Streamlit, Requests, Twilio
  * **Core**: Python, NumPy, psutil

-----

##  Getting Started

### Prerequisites

  * Python 3.9+
  * FFmpeg (accessible in your system's PATH)
  * Git

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2\. Set Up Environment

It is highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3\. Install Dependencies

Install all required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

For high-performance RTSP streaming, this project uses `vidgear`. You can install it with the provided script:

```bash
chmod +x install_vidgear.sh
./install_vidgear.sh
```

### 4\. Configure Environment Variables

Create a `.env` file in the root directory of the project. This file is listed in `.gitignore` and will not be committed to your repository.

```env
# Google Gemini API Key
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

# --- For WhatsApp Alerts via Meta API (used in ResNet_AnomalywithWA.py) ---
META_ACCESS_TOKEN="YOUR_META_PERMANENT_ACCESS_TOKEN"
WHATSAPP_PHONE_ID="YOUR_WHATSAPP_BUSINESS_PHONE_NUMBER_ID"
RECIPIENT_PHONE_NUMBER="RECIPIENT_PHONE_NUMBER_WITH_COUNTRY_CODE"

# --- For WhatsApp Alerts via Twilio (used in videoInput.py) ---
TWILIO_ACCOUNT_SID="YOUR_TWILIO_ACCOUNT_SID"
TWILIO_AUTH_TOKEN="YOUR_TWILIO_AUTH_TOKEN"
TWILIO_PHONE_NUMBER="YOUR_TWILIO_WHATSAPP_NUMBER"
# RECIPIENT_PHONE_NUMBER is shared
```

-----

##  How to Run

### 1\. Video Search & Retrieval System

#### Step 1: Index Your Videos

Run the `indexing.py` script to process your videos and create the searchable FAISS index. You will be prompted to enter the directory containing your videos.

```bash
python VideoDescripter+QueryRetrieval/indexing.py
```

This will create `video_library.faiss` and `video_library_metadata.json` in the project root. The script can be run multiple times; it will automatically skip videos that have already been indexed.

#### Step 2: Launch the Search UI

Start the Streamlit web application to search your indexed library.

```bash
streamlit run VideoDescripter+QueryRetrieval/app.py
```

### 2\. Live Anomaly Detection

This pipeline takes a live RTSP stream or a local video file as input and monitors for anomalies.

```bash
python LiveAnomalyDetection/ResNet_AnomalywithWA.py
```

The script will prompt you to choose between an RTSP stream or a video file and ask for the corresponding path/URL.

### 3\. Person Re-Identification

This script runs person detection and re-identification on a video file.

**Note:** This script requires a pre-trained model file named `best_model.pt` in the `LiveAnomalyDetection` directory.

```bash
python LiveAnomalyDetection/ReID_VideoInput.py
```

-----

## Configuration

Several key parameters can be adjusted directly within the Python scripts:

  * In `LiveAnomalyDetection/ResNet_AnomalywithWA.py`:
      * `BATCH_SIZE`: Number of frames to process for one anomaly check.
      * `TARGET_FPS`: The desired frames per second to process from the source.
      * `VLM_TRIGGER_INTERVAL`: Number of consecutive anomaly frames required to trigger a deeper Gemini analysis.
  * In `VideoDescripter+QueryRetrieval/indexing.py`:
      * `FPS`: The frames per second to sample from videos during indexing.
      * `BATCH_SIZE`: Number of frames to send to Gemini in a single API call.
