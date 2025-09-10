# === Core Libraries ===
import cv2
import time
import os
import numpy as np
from PIL import Image

# === ML & AI Libraries ===
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import faiss
import timm

class APN_Model(nn.Module):
    def __init__(self, emb_size=512):
        super(APN_Model, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Linear(
            in_features=self.efficientnet.classifier.in_features,
            out_features=emb_size
        )

    def forward(self, images):
        embeddings = self.efficientnet(images)
        return embeddings


class ReIdSystem:
    """
    Manages person detection (YOLOv8) and feature extraction using your custom model.
    """
    def __init__(self, model_weights_path, yolo_model_name='yolov8n.pt', device='cpu'):
        print(" Loading AI models...")
        self.device = torch.device(device)
        print(f"   - Loading YOLOv8 model: {yolo_model_name}")
        self.detector = YOLO(yolo_model_name)
        print(f"   - Loading your custom Siamese model from: {model_weights_path}")
        self.reid_model = APN_Model()
        self.reid_model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.reid_model.to(self.device)
        self.reid_model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(" All models loaded successfully.")

    @torch.no_grad()
    def process_frame(self, frame, detection_confidence=0.5, imgsz=640, nms_iou=0.7):
        """Detects people and extracts feature vectors for each."""
        results = self.detector(
            frame,
            classes=[0],
            conf=detection_confidence,
            imgsz=imgsz,
            iou=nms_iou,
            verbose=False,
            device=self.device
        )

        person_details = []
        for result in results:
            for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                cropped_person_cv2 = frame[y1:y2, x1:x2]
                if cropped_person_cv2.size == 0: continue

                pil_image = Image.fromarray(cv2.cvtColor(cropped_person_cv2, cv2.COLOR_BGR2RGB))
                img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                feature_vector = self.reid_model(img_tensor).cpu().numpy().flatten()

                person_details.append({
                    'box': box,
                    'score': score,
                    'vector': feature_vector
                })
        return person_details


class VectorDatabase:
    def __init__(self, feature_dim=512, similarity_threshold=0.75, smoothing_alpha=0.2):
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(feature_dim))
        self.next_person_id = 1
        self.threshold = similarity_threshold
        self.alpha = smoothing_alpha
        self.stored_vectors = {}
        print(f"Vector Database (Faiss) initialized for {feature_dim}-D vectors.")

    def search_and_identify(self, feature_vector):
        vector_norm = np.array([feature_vector], dtype=np.float32)
        faiss.normalize_L2(vector_norm)

        if self.index.ntotal == 0:
            return self._add_person(feature_vector, vector_norm)

        distances, ids = self.index.search(vector_norm, 1)

        if ids.size > 0 and distances[0][0] > self.threshold:
            person_id = ids[0][0]
            old_vector = self.stored_vectors[person_id]
            new_smoothed_vector = (self.alpha * feature_vector) + ((1 - self.alpha) * old_vector)
            self.stored_vectors[person_id] = new_smoothed_vector
            vector_to_add_norm = np.array([new_smoothed_vector], dtype=np.float32)
            faiss.normalize_L2(vector_to_add_norm)
            self.index.add_with_ids(vector_to_add_norm, np.array([person_id]))
            return person_id
        else:
            return self._add_person(feature_vector, vector_norm)

    def _add_person(self, original_vector, normalized_vector):
        """Adds a new person to the database."""
        person_id = self.next_person_id
        self.index.add_with_ids(normalized_vector, np.array([person_id]))
        self.stored_vectors[person_id] = original_vector
        print(f"➕ New Person Detected. Assigning ID: Person_{person_id}")
        self.next_person_id += 1
        return person_id


def main():
    YOLO_MODEL_NAME = 'yolov8n.pt'
    DETECTION_CONFIDENCE = 0.5
    DETECTION_IMG_SIZE = 640
    NMS_IOU_THRESHOLD = 0.7

    # --- Re-Identification ---
    MODEL_WEIGHTS_PATH = "best_model.pt"
    REID_EMBEDDING_DIM = 512
    REID_SIMILARITY_THRESHOLD = 0.7
    REID_SMOOTHING_ALPHA = 0.1

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Model weights not found at: {MODEL_WEIGHTS_PATH}"); return

    try:
        reid_system = ReIdSystem(
            model_weights_path=MODEL_WEIGHTS_PATH,
            yolo_model_name=YOLO_MODEL_NAME,
            device='cpu'
        )
        vector_db = VectorDatabase(
            feature_dim=REID_EMBEDDING_DIM,
            similarity_threshold=REID_SIMILARITY_THRESHOLD,
            smoothing_alpha=REID_SMOOTHING_ALPHA
        )
    except Exception as e:
        print(f"Critical setup error: {e}"); return

    video_path = r"C:\Users\Aaarat\PycharmProjects\SIH_LiveCam\Abuse001_x264.mp4\Abuse001_x264.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}"); return

    cap = cv2.VideoCapture(video_path)
    print("\nLive feed starting. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video."); break

        persons_in_frame = reid_system.process_frame(
            frame,
            detection_confidence=DETECTION_CONFIDENCE,
            imgsz=DETECTION_IMG_SIZE,
            nms_iou=NMS_IOU_THRESHOLD
        )

        for person in persons_in_frame:
            person_id = vector_db.search_and_identify(person['vector'])
            x1, y1, x2, y2 = map(int, person['box'])
            label = f"Person_{person_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 100, 0), 2)

        cv2.imshow('Custom Person Re-Identification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Pipeline shutdown complete.")


if __name__ == "__main__":
    main()