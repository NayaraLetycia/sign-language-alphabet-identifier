# live_inference.py
# Detecção de mão com MediaPipe, recorte da ROI e classificação por letra (A–Z).
# Carrega modelo e classes salvos em artifacts/.

import os
import json
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import mediapipe as mp


# === CONFIG ===
MODEL_PATH = "artifacts/best_resnet18.pt"
CLASSES_PATH = "artifacts/classes.json"
IMG_SIZE = 224
VOTE_WINDOW = 7   # suavização temporal via voto majoritário

# === PREPROCESS ===
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# === CARREGAR MODELO E CLASSES ===
assert os.path.exists(MODEL_PATH), f"Modelo não encontrado em {MODEL_PATH}"
assert os.path.exists(CLASSES_PATH), f"Arquivo de classes não encontrado em {CLASSES_PATH}"

ckpt = torch.load(MODEL_PATH, map_location="cpu")
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(ckpt["model"])
model = model.to(device).eval()

# === MEDIAPIPE HANDS ===
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def crop_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    xs, ys = [], []
    for lm in hand_landmarks.landmark:
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))
    pad = 20
    x1, y1 = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
    x2, y2 = min(max(xs) + pad, w), min(max(ys) + pad, h)
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)

def predict_letter(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    x = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        idx = int(np.argmax(prob))
        return class_names[idx], float(prob[idx])

def main():
    # Tente outras câmeras: 1, 2...
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vote_buf = deque(maxlen=VOTE_WINDOW)

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Não foi possível acessar a câmera. Tente outro índice (1/2).")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(frame_rgb)

            pred_text = "..."
            if res.multi_hand_landmarks:
                for handLms in res.multi_hand_landmarks:
                    roi, (x1,y1,x2,y2) = crop_hand(frame, handLms)
                    if roi is not None and roi.size > 0:
                        out = predict_letter(roi)
                        if out:
                            letter, conf = out
                            vote_buf.append(letter)
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                            cv2.putText(frame, f"{letter} ({conf:.2f})", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                if len(vote_buf) > 0:
                    pred_text = Counter(vote_buf).most_common(1)[0][0]

            cv2.putText(frame, f"Pred: {pred_text}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
            cv2.imshow("Leitor de Libras (A–Z)", frame)

            # ESC para sair
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
