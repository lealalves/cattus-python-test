import cv2
import base64
from ultralytics import YOLO

# Carrega o modelo treinado
model = YOLO("best.pt")  # Substitua pelo seu modelo treinado

def get_frame():
    cap = cv2.VideoCapture(0)  # Captura da câmera

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Faz a detecção
        results = model(frame)

        # Converte o frame para base64 para enviar via WebSocket
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        yield {"image": frame_base64, "detections": detections}  # Retorna o frame e detecções

    cap.release()
