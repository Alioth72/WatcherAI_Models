import google.generativeai as genai
import cv2
import os
import queue
import threading
import base64
import json
import datetime
from dotenv import load_dotenv
from tqdm import tqdm

# --- GLOBAL CONFIG ---
FPS = 5                # Global FPS for sampling
BATCH_SIZE = 100        # Frames per batch
FRAME_INTERVAL = 10     # Take every 10th frame in a batch for Gemini
MODEL_NAME = 'gemini-2.5-pro'

# Queue to hold frames
frame_queue = queue.Queue()

# --- GEMINI SETUP ---
def setup_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")
    genai.configure(api_key=api_key)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    return genai.GenerativeModel(MODEL_NAME, safety_settings=safety_settings)

# --- FRAME CAPTURE THREAD ---
def capture_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(input_fps // FPS) if input_fps > FPS else 1
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            ts_sec = frame_count / input_fps
            timestamp = f"{int(ts_sec//3600):02d}:{int((ts_sec%3600)//60):02d}:{int(ts_sec%60):02d}"
            frame_queue.put({"timestamp": timestamp, "frame": frame})
        
        frame_count += 1

    cap.release()
    frame_queue.put(None)  # Signal end of frames
    print("Finished capturing frames.")

# --- GEMINI PROMPT PROCESSOR ---
def analyze_batch(frames, model, log_file):
    # Select every 10th frame
    selected_frames = frames[::FRAME_INTERVAL]
    
    # Convert to Base64
    images_base64 = []
    for f in selected_frames:
        _, buffer = cv2.imencode('.jpg', f["frame"])
        encoded = base64.b64encode(buffer).decode('utf-8')
        images_base64.append({"mime_type": "image/jpeg", "data": encoded})
    
    # Structured prompt
    prompt = """
Analyze this set of images for forensic-level scene understanding.

Return JSON ONLY:
{
  "overall_scene": {
    "location": "",
    "time_of_day": "",
    "people_count": 0,
    "objects_detected": [],
    "activity_summary": ""
  },
  "suspicious_events": [
    {
      "timestamp": "",
      "description": "",
      "actors": [],
      "objects": []
    }
  ]
}
"""

    content = [prompt] + images_base64
    
    try:
        response = model.generate_content(content)
        data = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        data["frame_batch_start"] = frames[0]["timestamp"]
        data["frame_batch_end"] = frames[-1]["timestamp"]

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data) + "\n")

        return data
    except Exception as e:
        print(f"Error analyzing batch: {e}")
        return None

# --- MAIN PIPELINE ---
def main():
    video_path = input("Enter path to the video: ").strip()
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"forensic_log_{timestamp_str}.jsonl"
    
    model = setup_gemini()
    threading.Thread(target=capture_frames, args=(video_path,), daemon=True).start()

    batch = []
    print("Processing video...")

    while True:
        item = frame_queue.get()
        if item is None:
            if batch:
                analyze_batch(batch, model, log_file)
            break
        
        batch.append(item)
        if len(batch) == BATCH_SIZE:
            analyze_batch(batch, model, log_file)
            batch = []

    print(f"Processing complete. Logs saved in {log_file}")

if __name__ == "__main__":
    main()
