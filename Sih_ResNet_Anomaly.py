import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras import mixed_precision
from datetime import datetime

mixed_precision.set_global_policy('mixed_float16')


class FrameData:
    def __init__(self, frame: np.ndarray, timestamp: datetime):
        self.frame = frame
        self.timestamp = timestamp

print(" Loading anomaly detection models (ResNet50 + SVM)...")
try:
    feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    svm_path = "svm_model.pkl"
    svm_model = joblib.load(svm_path)
    print("Warming up GPU...")
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float16)
    feature_extractor.predict(dummy_input, verbose=0)
    print(" Anomaly detection models loaded and ready.")

except Exception as e:
    print(f"Critical error loading models: {e}")
    # Exit if models can't be loaded, as the script is useless without them.
    exit()


def process_batch(frame_batch: list[FrameData]) -> bool:
    """
    Processes a batch of frames to detect anomalies.

    Args:
        frame_batch: A list of FrameData objects.

    Returns:
        bool: True if an anomaly is detected in any frame, False otherwise.
    """
    if not frame_batch:
        return False

    anomaly_found_in_batch = False
    frames = [fd.frame for fd in frame_batch]

    # --- 2. PREPROCESSING ---
    preprocessed_batch = np.array(
        [tf.keras.applications.resnet50.preprocess_input(image.img_to_array(f)) for f in frames],
        dtype=np.float16
    )
    features = feature_extractor.predict(preprocessed_batch, verbose=0)

    predictions = svm_model.predict(features)
    for pred in predictions:
        if pred == 1:
            anomaly_found_in_batch = True
            break
    return anomaly_found_in_batch