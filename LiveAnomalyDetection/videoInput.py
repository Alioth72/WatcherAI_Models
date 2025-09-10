import cv2
import time
import signal
import threading
import os
import queue
import base64
import json
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait

# === VLM, Environment & Messaging Libraries ===
import google.generativeai as genai
from dotenv import load_dotenv
from twilio.rest import Client

# === Your Custom Modules ===
from Sih_ResNet_Anomaly import process_batch
from rtspHandler import RTSPFrameCapture

try:
    ffmpeg_bin_path = r"E:\ffmpeg-2025-09-08-git-45db6945e9-full_build\bin"
    os.environ['PATH'] = ffmpeg_bin_path + os.pathsep + os.environ.get('PATH', '')
    print("FFmpeg path configured successfully for this session.")
except Exception as e:
    print(f" Error configuring FFmpeg path: {e}")

executor = ThreadPoolExecutor(max_workers=3)  # Increased workers for VLM + Twilio
shutdown_event = threading.Event()
BATCH_SIZE = 100  # Frames per batch for anomaly detection
TARGET_FPS = 5  # Capture frames per second

# --- VLM Trigger Config ---
VLM_TRIGGER_INTERVAL = 50  # Trigger VLM for every 50 continuous anomaly frames
VLM_COOLDOWN_SECONDS = 15  # Wait 15s after a VLM call

# --- Gemini VLM Config ---
FRAME_INTERVAL_FOR_GEMINI = 10  # Take every 10th frame in a batch for Gemini
MODEL_NAME = 'gemini-2.5-flash'


# === Twilio & VLM Setup ===
def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    print(f"Gemini Model '{MODEL_NAME}' configured.")
    return genai.GenerativeModel(MODEL_NAME)


def send_whatsapp_alert(anomaly_data):
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    twilio_number = os.getenv('TWILIO_PHONE_NUMBER')
    recipient_number = os.getenv('RECIPIENT_PHONE_NUMBER')

    if not all([account_sid, auth_token, twilio_number, recipient_number]):
        print("Twilio credentials not fully configured in .env file. Skipping WhatsApp alert.")
        return

    try:
        client = Client(account_sid, auth_token)
        desc = anomaly_data.get('anomaly_description', {})

        # Format a clean, readable message
        message_body = (
            f"🚨ALERT ALERT !* 🚨\n\n"
            f"*Summary:* {desc.get('summary', 'N/A')}\n"
            f"*Confidence:* {desc.get('confidence_level', 'N/A')}\n\n"
            f"*Details:*\n"
        )
        for action in desc.get('involved_persons_actions', []):
            message_body += f"- {action}\n"

        message_body += f"\n*Timeframe:* {anomaly_data.get('batch_start_timestamp', 'N/A')}"

        message = client.messages.create(
            from_=f'whatsapp:{twilio_number}',
            body=message_body,
            to=f'whatsapp:{recipient_number}'
        )
        print(f"📲 WhatsApp alert sent successfully! SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send WhatsApp alert: {e}")


def analyze_and_alert(frames, model, log_file):
    print("🔬 Submitting batch for Gemini analysis...")
    anomaly_data = analyze_anomaly_with_gemini(frames, model, log_file)
    if anomaly_data:
        send_whatsapp_alert(anomaly_data)


def analyze_anomaly_with_gemini(frames, model, log_file):
    if not frames:
        return None

    selected_frames = frames[::FRAME_INTERVAL_FOR_GEMINI]
    images_base64 = []
    for f in selected_frames:
        _, buffer = cv2.imencode('.jpg', f.frame)
        encoded = base64.b64encode(buffer).decode('utf-8')
        images_base64.append({"mime_type": "image/jpeg", "data": encoded})

    prompt = """
A machine learning model has flagged this sequence of frames for a potential anomaly or suspicious activity. 
Your task is to act as a forensic analyst and provide a concise, factual description of the events.
Focus on describing the specific actions that are likely anomalous. What is happening that is out of the ordinary?
Return a single JSON object ONLY. Do not include ```json or any other text.
{
  "anomaly_description": {
    "summary": "A brief, one-sentence summary of the anomalous event.",
    "involved_persons_actions": ["Describe the actions of person 1."],
    "involved_objects": ["object1", "object2"],
    "confidence_level": "High/Medium/Low based on the visual evidence."
  }
}
"""
    content = [prompt] + images_base64

    try:
        response = model.generate_content(content)
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned_text)
        data["batch_start_timestamp"] = frames[0].timestamp.isoformat()
        data["batch_end_timestamp"] = frames[-1].timestamp.isoformat()
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data) + "\n")
        print("Gemini analysis complete. Logged to file.")
        return data
    except Exception as e:
        print(f"Error during Gemini analysis: {e}")
        return None

def signal_handler(signum, frame):
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class FrameData:
    def __init__(self, frame, timestamp: datetime):
        self.frame = frame
        self.timestamp = timestamp


def run_pipeline(source, gemini_model, log_file, is_rtsp=False):
    """
    Main pipeline to capture frames, detect anomalies, and trigger VLM analysis.
    Works for both video files and RTSP streams.
    """
    # --- State Tracking Variables ---
    consecutive_anomaly_frames = 0
    last_vlm_trigger_frame_count = 0
    vlm_cooldown_until = 0
    futures = []
    queue_frames = deque()

    # --- Capture Setup ---
    if is_rtsp:
        capture = RTSPFrameCapture(source, required_fps=TARGET_FPS, camera_name="RTSP_Camera")
        if not capture.start(): return
        print("Waiting for RTSP stream to initialize...")
        time.sleep(3)
    else:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            print(f"Unable to open video: {source}")
            return
        # <<< START: Frame Skipping Logic >>>
        input_fps = capture.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(input_fps / TARGET_FPS))
        print(
            f"📹 Video file detected. Input FPS: {input_fps:.2f}. Processing 1 frame every {frame_skip} frames to achieve ~{TARGET_FPS} FPS.")
        frame_count = 0

    try:
        while not shutdown_event.is_set():
            if is_rtsp:
                frame = capture.get_frame()
                if frame is None:
                    time.sleep(1 / (TARGET_FPS * 2))
                    continue
                frame_to_process = frame
            else:  # Is a video file
                ret, frame = capture.read()
                if not ret:
                    break

                frame_to_process = None  # Default to no frame
                if frame_count % frame_skip == 0:
                    frame_to_process = frame
                frame_count += 1

            # --- Frame Processing ---
            if frame_to_process is not None:
                frame_resized = cv2.resize(frame_to_process, (224, 224))
                queue_frames.append(FrameData(frame_resized, datetime.now()))

            # --- Batch Processing and Anomaly Detection ---
            if len(queue_frames) >= BATCH_SIZE:
                current_batch = [queue_frames.popleft() for _ in range(BATCH_SIZE)]

                is_anomalous = process_batch(current_batch)

                if is_anomalous:
                    consecutive_anomaly_frames += len(current_batch)
                    print(f"Anomaly detected! Consecutive anomaly frame count: {consecutive_anomaly_frames}")
                else:
                    consecutive_anomaly_frames = 0
                    last_vlm_trigger_frame_count = 0

                # --- VLM Trigger Logic ---
                frames_since_last_trigger = consecutive_anomaly_frames - last_vlm_trigger_frame_count

                if frames_since_last_trigger >= VLM_TRIGGER_INTERVAL and time.time() > vlm_cooldown_until:
                    print(f"Anomaly has persisted for {frames_since_last_trigger} more frames.")
                    future = executor.submit(analyze_and_alert, current_batch, gemini_model, log_file)
                    futures.append(future)

                    last_vlm_trigger_frame_count = consecutive_anomaly_frames
                    vlm_cooldown_until = time.time() + VLM_COOLDOWN_SECONDS
                    print(
                        f"VLM triggered. Cooldown until {datetime.fromtimestamp(vlm_cooldown_until).strftime('%H:%M:%S')}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if is_rtsp:
            capture.stop()
        else:
            capture.release()
        print("Waiting for all background tasks to finish...")
        wait(futures)
        executor.shutdown(wait=True)
        print(" Pipeline shutdown complete.")


def main():
    load_dotenv()  # Load all environment variables at the start
    try:
        gemini_model = setup_gemini()
    except ValueError as e:
        print(f"{e}")
        return

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"forensic_log_{timestamp_str}.jsonl"
    print(f" VLM analysis will be logged to: {log_file}")

    print("\nChoose Input Source:")
    print("1. RTSP Stream\n2. Video File")
    choice = input("Enter choice (1/2): ")

    if choice == "1":
        source_path = input("Enter RTSP stream URL: ")
        run_pipeline(source_path, gemini_model, log_file, is_rtsp=True)
    elif choice == "2":
        source_path = input("Enter video file path: ")
        if not os.path.exists(source_path):
            print(f"Video not found: {source_path}")
            return
        run_pipeline(source_path, gemini_model, log_file, is_rtsp=False)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()