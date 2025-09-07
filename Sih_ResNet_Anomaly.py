import time
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
import joblib
import tensorflow as tf
from tensorflow.keras import mixed_precision
from collections import deque

mixed_precision.set_global_policy('mixed_float16')

# --- Paths ---
video_path = r"Abuse001_x264.mp4\Abuse001_x264.mp4"
svm_path = "svm_model.pkl"

# --- Load models ---
model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
svm_model = joblib.load(svm_path)

# --- Warm-up GPU ---
dummy = np.zeros((1, 224, 224, 3), dtype=np.float16)
model.predict(dummy)





# --- Functions ---
def process_batch(frameBatch):
    """
    Process a batch of FrameData objects:
    1. Extract frames from FrameData
    2. Preprocess frames for ResNet
    3. Extract features
    4. Predict with SVM
    5. Print predictions with timestamps
    """
    if not frameBatch:
        print("⚠️ Empty batch received, skipping...")
        return

    # Extract raw frames and timestamps
    frames = [fd.frame for fd in frameBatch]
    timestamps = [fd.timestamp for fd in frameBatch]

    # Preprocess frames
    batch = np.array(
        [tf.keras.applications.resnet50.preprocess_input(image.img_to_array(f)) for f in frames],
        dtype=np.float16
    )

    # Extract features
    feats = model.predict(batch, verbose=0)

    # Predict with SVM
    preds = svm_model.predict(feats)

    # Display results
    for ts, pred in zip(timestamps, preds):
        print(f"[{ts}] Prediction: {pred}")

    print(f"✅ Processed batch of {len(frames)} frames.\n")



# def process_batch(frames, frame_indices):
#     """Run ResNet + SVM on a batch of frames."""
#     if not frames:
#         return
#     batch = np.array([tf.keras.applications.resnet50.preprocess_input(image.img_to_array(f)) for f in frames], dtype=np.float16)
#     feats = model.predict(batch, verbose=0)
#     preds = svm_model.predict(feats)
#     for idx, pred in zip(frame_indices, preds):
#         print(f"Frame {idx}: Pred={pred}")

# --- Pipeline ---
# target_fps = 5   # only capture 5 frames per second
# batch_size = 100
# queue = deque()
# queue_indices = []

# cap = cv2.VideoCapture(video_path)
# total_start = time.time()
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# video_fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"Original video length: {total_frames/video_fps:.2f}s, FPS: {video_fps}, Total frames: {total_frames}")

# frame_idx = 0
# step = int(video_fps / target_fps)  # number of frames to skip to achieve target_fps

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # only process frames according to step
#     if frame_idx % step == 0:
#         frame_resized = cv2.resize(frame, (224, 224))
#         queue.append(frame_resized)
#         queue_indices.append(frame_idx)

#         # Process batch when queue is full
#         if len(queue) == batch_size:
#             batch_start = time.time()
#             process_batch(list(queue), list(queue_indices))
#             batch_end = time.time()
#             print(f"Processed batch of {batch_size} frames in {batch_end - batch_start:.2f}s")
#             queue.clear()
#             queue_indices.clear()

#     frame_idx += 1

# # Process any remaining frames
# if queue:
#     batch_start = time.time()
#     process_batch(list(queue), list(queue_indices))
#     batch_end = time.time()
#     print(f"Processed final batch of {len(queue)} frames in {batch_end - batch_start:.2f}s")

# cap.release()
# total_end = time.time()
# print(f"Total processing time for video: {total_end - total_start:.2f}s")
