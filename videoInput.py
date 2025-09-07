import cv2
import time
import signal
import threading
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from Sih_ResNet_Anomaly import process_batch
from rtspHandler import RTSPFrameCapture
 # <-- import your FrameData class

# === Global Settings ===
executor = ThreadPoolExecutor(max_workers=2)
shutdown_event = threading.Event()
BATCH_SIZE = 100   # Frames per batch
TARGET_FPS = 5     # Capture 5 frames per second


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\n🛑 Received shutdown signal {signum}. Exiting gracefully...")
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class FrameData:
    def __init__(self, frame, timestamp: datetime):
        """
        FrameData holds a frame (numpy array) and its timestamp
        """
        self.frame = frame
        self.timestamp = timestamp

    def __repr__(self):
        return f"FrameData(timestamp={self.timestamp})"
    
def process_batch_async(frame_data_list):
    """Submit frame batch for asynchronous processing."""
    return executor.submit(process_batch, frame_data_list)


def run_video_pipeline(video_path):
    """Capture frames from a video file and send batches to process_batch."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Unable to open video: {video_path}")
        return

    total_start = time.time()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video length: {total_frames/video_fps:.2f}s, FPS: {video_fps}, Frames: {total_frames}")

    frame_idx = 0
    step = max(1, int(video_fps / TARGET_FPS))
    queue_frames = deque()
    futures = []  # store all batch futures

    try:
        while not shutdown_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames based on target FPS
            if frame_idx % step == 0:
                frame_resized = cv2.resize(frame, (224, 224))
                frame_obj = FrameData(frame_resized, datetime.now())
                queue_frames.append(frame_obj)

                # Process batch when we reach 100 frames
                if len(queue_frames) == BATCH_SIZE:
                    print(f"🚀 Submitting Video batch of {BATCH_SIZE} frames")
                    futures.append(process_batch_async(list(queue_frames)))
                    queue_frames.clear()

            frame_idx += 1

        # Final leftover frames
        if queue_frames:
            print(f"🚀 Submitting final Video batch of {len(queue_frames)} frames")
            futures.append(process_batch_async(list(queue_frames)))

        cap.release()
        print("⏳ Waiting for all anomaly detection tasks to finish...")
        wait(futures)
        total_end = time.time()
        print(f"✅ Total processing time: {total_end - total_start:.2f}s")

    except KeyboardInterrupt:
        print("\nStopped by user.")


def run_rtsp_pipeline(rtsp_url):
    """Capture frames from an RTSP stream and send batches to process_batch."""
    rtsp_capture = RTSPFrameCapture(rtsp_url, required_fps=TARGET_FPS, camera_name="RTSP_Camera")
    if not rtsp_capture.start():
        print("❌ Failed to start RTSP stream")
        return

    print("Waiting for RTSP stream to initialize...")
    time.sleep(3)

    frame_idx = 0
    queue_frames = deque()
    futures = []

    try:
        while not shutdown_event.is_set():
            frame = rtsp_capture.get_frame()
            if frame is not None:
                frame_resized = cv2.resize(frame, (224, 224))
                frame_obj = FrameData(frame_resized, datetime.now())
                queue_frames.append(frame_obj)

                # Process batch when full
                if len(queue_frames) == BATCH_SIZE:
                    print(f"🚀 Submitting RTSP batch of {BATCH_SIZE} frames")
                    futures.append(process_batch_async(list(queue_frames)))
                    queue_frames.clear()

                frame_idx += 1
            else:
                time.sleep(0.1)

        # Final leftover frames
        if queue_frames:
            print(f"🚀 Submitting final RTSP batch of {len(queue_frames)} frames")
            futures.append(process_batch_async(list(queue_frames)))

        print("⏳ Waiting for all anomaly detection tasks to finish...")
        wait(futures)
        print("✅ RTSP pipeline completed.")

    except KeyboardInterrupt:
        print("\nStopped by user.")


def main():
    print("Choose Input Source:")
    print("1. RTSP Stream")
    print("2. Video File")
    choice = input("Enter choice (1/2): ")

    if choice == "1":
        rtsp_url = input("Enter RTSP stream URL: ")
        run_rtsp_pipeline(rtsp_url)
    elif choice == "2":
        video_path = input("Enter video file path: ")
        run_video_pipeline(video_path)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
