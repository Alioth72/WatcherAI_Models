# File: app.py
import streamlit as st
import cv2
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
import ffmpeg

# (Configuration and cached functions remain the same)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
FAISS_INDEX_PATH = "video_library.faiss"
METADATA_PATH = "video_library_metadata.json"

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)
@st.cache_resource
def load_index_and_metadata():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return None, None
    with st.spinner("Loading search index for the first time..."):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    return index, metadata

# (Visualization functions get_key_frames and extract_clip remain the same)
def get_key_frames(video_path, start_frame, end_frame):
    # ... (code is unchanged)
    cap = cv2.VideoCapture(video_path)
    frames = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if ret: frames['start'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if end_frame - start_frame > 1:
        mid_frame_num = start_frame + (end_frame - start_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_num)
        ret, frame = cap.read()
        if ret: frames['middle'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
    ret, frame = cap.read()
    if ret: frames['end'] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return frames
    
def extract_clip(video_path, start_frame, end_frame):
    # ... (code is unchanged)
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        start_time_sec = start_frame / fps
        duration_sec = (end_frame - start_frame) / fps
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_filename = temp_file.name
        (
            ffmpeg
            .input(video_path, ss=start_time_sec)
            .output(temp_filename, t=duration_sec, vcodec='libx264', acodec='aac', strict='experimental')
            .overwrite_output()
            .run(quiet=True)
        )
        return temp_filename
    except Exception as e:
        st.error(f"Failed to extract clip with FFmpeg: {e}.")
        return None

# --- STREAMLIT UI (MODIFIED) ---
st.set_page_config(page_title="VLM Video Search", layout="wide")
st.title("👁️ AI-Powered Video Library Search")

index, metadata_store = None, None

if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
    st.error("Index files not found! Please run the `process_videos.py` script first to analyze your videos.")
else:
    query = st.text_input("Enter your query:", placeholder="e.g., a person in a red shirt OR an event around 2023-11-20 15:30:00")

    if st.button("Search"):
        if query:
            embedder = load_embedder()
            index, metadata_store = load_index_and_metadata()
            
            if index is None:
                st.error("Index files could not be loaded. Please re-run `process_videos.py`.")
            else:
                with st.spinner("Searching for relevant segments..."):
                    CANDIDATE_POOL_SIZE = 20
                    TARGET_RESULTS = 3
                    query_embedding = embedder.encode([query])
                    distances, indices = index.search(query_embedding, CANDIDATE_POOL_SIZE)
                    
                    unique_results = []
                    seen_video_paths = set()
                    
                    if indices[0].size and indices[0][0] != -1:
                        for i, idx in enumerate(indices[0]):
                            if idx == -1: continue
                            metadata = metadata_store[idx]
                            video_path = metadata.get('video_path')
                            if video_path and video_path not in seen_video_paths:
                                unique_results.append(metadata)
                                seen_video_paths.add(video_path)
                            if len(unique_results) >= TARGET_RESULTS:
                                break
                    
                    st.subheader("Search Results")
                    if not unique_results:
                        st.write("No relevant segments found.")
                    else:
                        for i, metadata in enumerate(unique_results):
                            full_video_path = metadata['video_path']
                            st.markdown("---")
                            
                            st.subheader(f"Rank {i+1}: Best match in `{os.path.basename(full_video_path)}`")
                            
                            # --- NEW: Display Absolute Timestamps ---
                            st.info(f"**Event Time (Real World):** From **{metadata['absolute_start_time']}** to **{metadata['absolute_end_time']}**")

                            with st.spinner(f"Extracting visuals for Rank {i+1}..."):
                                # (Visual extraction and display logic remains the same)
                                if not os.path.exists(full_video_path):
                                    st.error(f"Source video not found: {full_video_path}")
                                    continue
                                key_frames = get_key_frames(full_video_path, metadata['start_frame'], metadata['end_frame'])
                                if 'middle' in key_frames:
                                    kf1, kf2, kf3 = st.columns(3)
                                    with kf1: st.image(key_frames['start'], caption="Start Frame", use_container_width=True)
                                    with kf2: st.image(key_frames['middle'], caption="Middle Frame", use_container_width=True)
                                    with kf3: st.image(key_frames['end'], caption="End Frame", use_container_width=True)
                                else:
                                    kf1, kf2 = st.columns(2)
                                    with kf1: st.image(key_frames.get('start'), caption="Start Frame", use_container_width=True)
                                    with kf2: st.image(key_frames.get('end'), caption="End Frame", use_container_width=True)
                                clip_path = extract_clip(full_video_path, metadata['start_frame'], metadata['end_frame'])
                                if clip_path:
                                    st.video(clip_path)
                                    os.remove(clip_path)
        else:
            st.warning("Please enter a query.")