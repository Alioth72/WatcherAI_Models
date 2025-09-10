# File: process_videos.py
import google.generativeai as genai
import cv2
import os
import queue
import threading
import base64
import json
import datetime
import time
from dotenv import load_dotenv
from tqdm import tqdm
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
FPS = 5
BATCH_SIZE = 100
FRAME_INTERVAL = 10
MODEL_NAME = 'gemini-1.5-pro-latest'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "video_library.faiss"
METADATA_PATH = "video_library_metadata.json"

frame_queue = queue.Queue(maxsize=BATCH_SIZE*2)

# (setup_gemini, setup_embedder, capture_frames functions are the same as before)
def setup_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY not found.")
    genai.configure(api_key=api_key)
    safety_settings = [{"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_NONE"} for c in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]]
    generation_config = {"response_mime_type": "application/json"}
    return genai.GenerativeModel(MODEL_NAME, safety_settings=safety_settings, generation_config=generation_config)

def setup_embedder():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedder is ready.")
    return embedder

def capture_frames(video_path):
    # ... (code is unchanged)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(input_fps // FPS) if input_fps > FPS and FPS > 0 else 1
    for frame_count in tqdm(range(total_frames), desc=f"Capturing from {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret: break
        if frame_count % frame_skip == 0:
            ts_sec = frame_count / input_fps
            timestamp = f"{int(ts_sec//3600):02d}:{int((ts_sec%3600)//60):02d}:{int(ts_sec%60):02d}"
            frame_queue.put({"timestamp": timestamp, "frame": frame, "frame_num": frame_count})
    cap.release()
    frame_queue.put(None)
    print(f"\nFinished capturing frames for {os.path.basename(video_path)}.")

def parse_time_string_to_timedelta(ts_str):
    # ... (code is unchanged)
    try:
        h, m, s = map(int, ts_str.split(':'))
        return datetime.timedelta(hours=h, minutes=m, seconds=s)
    except:
        return datetime.timedelta(seconds=0)

def analyze_and_prepare_batch(frames, model, embedder, video_path, video_start_datetime):
    # This function now returns the data instead of appending to global lists
    if not frames: return None, None
    
    selected_frames = frames[::FRAME_INTERVAL]
    images_base64 = []
    for f in selected_frames:
        _, buffer = cv2.imencode('.jpg', f["frame"])
        encoded = base64.b64encode(buffer).decode('utf-8')
        images_base64.append({"mime_type": "image/jpeg", "data": encoded})
    prompt = """Analyze this set of images... (rest of prompt is the same)"""
    content = [prompt] + images_base64
    try:
        response = model.generate_content(content)
        data = json.loads(response.text.strip())
        
        start_offset = parse_time_string_to_timedelta(frames[0]['timestamp'])
        absolute_start_time = video_start_datetime + start_offset
        absolute_end_time = video_start_datetime + parse_time_string_to_timedelta(frames[-1]['timestamp'])
        
        video_filename = os.path.basename(video_path)
        scene_data = data.get('overall_scene', {})
        
        embedding_text = f"The following event occurred in the video '{video_filename}' between {absolute_start_time.strftime('%Y-%m-%d %H:%M:%S')} and {absolute_end_time.strftime('%Y-%m-%d %H:%M:%S')}. ..."
        
        embedding = embedder.encode(embedding_text).tolist()
        
        metadata_entry = {
            "video_path": os.path.abspath(video_path),
            "start_time_offset": frames[0]['timestamp'],
            "end_time_offset": frames[-1]['timestamp'],
            "absolute_start_time": absolute_start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "absolute_end_time": absolute_end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "start_frame": frames[0]['frame_num'],
            "end_frame": frames[-1]['frame_num'],
            "document": embedding_text
        }
        return embedding, metadata_entry
    except Exception as e:
        print(f"\nError during analysis or preparation: {e}")
        return None, None

# --- NEW: Helper functions for incremental saving ---
def load_existing_data():
    """Loads existing index and metadata if they exist."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        print("Loading existing index and metadata...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        # Reconstruct embeddings from the FAISS index
        all_embeddings = [index.reconstruct(i).tolist() for i in range(index.ntotal)]
        with open(METADATA_PATH, 'r') as f:
            metadata_store = json.load(f)
        print(f"Loaded {len(metadata_store)} existing entries.")
        return all_embeddings, metadata_store
    return [], []

def save_index_and_metadata(all_embeddings, metadata_store):
    """Builds and saves the FAISS index and metadata file."""
    if not all_embeddings:
        print("No embeddings to save.")
        return
        
    print(f"\nRebuilding and saving index with {len(all_embeddings)} total entries...")
    embeddings_np = np.array(all_embeddings).astype('float32')
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)
    
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata_store, f)
    
    print(f"SUCCESS: Index saved to '{FAISS_INDEX_PATH}' and metadata to '{METADATA_PATH}'.")

# --- MODIFIED MAIN FUNCTION ---
def main():
    video_dir = input("Enter the path to the directory containing your videos: ").strip()
    if not os.path.isdir(video_dir):
        print(f"Error: Directory not found at '{video_dir}'")
        return

    model = setup_gemini()
    embedder = setup_embedder()
    
    # Load any data from previous runs
    all_embeddings, metadata_store = load_existing_data()
    
    # Create a set of already processed video paths for efficiency
    processed_videos = {item['video_path'] for item in metadata_store}
    
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]

    for video_filename in video_files:
        video_path = os.path.abspath(os.path.join(video_dir, video_filename))
        
        # Skip videos that are already in the index
        if video_path in processed_videos:
            print(f"\nSkipping '{video_filename}' as it is already in the index.")
            continue

        file_mod_time = os.path.getmtime(video_path)
        video_start_datetime = datetime.datetime.fromtimestamp(file_mod_time)
        print(f"\n--- Processing '{video_filename}' (Assumed Start Time: {video_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}) ---")
        
        # Temporary lists for the current video's data
        current_video_embeddings = []
        current_video_metadata = []

        while not frame_queue.empty(): frame_queue.get()
        capture_thread = threading.Thread(target=capture_frames, args=(video_path,), daemon=True)
        capture_thread.start()
        batch = []
        pbar = tqdm(desc=f"Processing Batches for {video_filename}")
        while capture_thread.is_alive() or not frame_queue.empty():
            try:
                item = frame_queue.get(timeout=1)
                if item is None: break
                batch.append(item)
                if len(batch) >= BATCH_SIZE:
                    embedding, metadata = analyze_and_prepare_batch(batch, model, embedder, video_path, video_start_datetime)
                    if embedding:
                        current_video_embeddings.append(embedding)
                        current_video_metadata.append(metadata)
                    batch = []
                    pbar.update(1)
                    print("\nBatch processed. Waiting for 31 seconds...")
                    time.sleep(31)
            except queue.Empty:
                continue
        if batch:
            embedding, metadata = analyze_and_prepare_batch(batch, model, embedder, video_path, video_start_datetime)
            if embedding:
                current_video_embeddings.append(embedding)
                current_video_metadata.append(metadata)
            pbar.update(1)
        pbar.close()
        
        # --- INCREMENTAL SAVE ---
        # Add the new data to the master lists and save everything
        if current_video_embeddings:
            all_embeddings.extend(current_video_embeddings)
            metadata_store.extend(current_video_metadata)
            save_index_and_metadata(all_embeddings, metadata_store)
        
        print(f"--- Finished processing and updated index for: {video_filename} ---")

    print("\nAll videos have been processed and indexed.")

if __name__ == "__main__":
    main()