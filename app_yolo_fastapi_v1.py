import cv2
import time
import queue
import torch
import uvicorn
import threading
from fastapi import FastAPI
from ultralytics import YOLO
from fastapi.responses import StreamingResponse

app = FastAPI()

# Carregar o modelo YOLO uma vez e enviá-lo para a GPU
model = YOLO("best.pt").to("cuda")

# Configurar a captura do vídeo
URL_CAMERA = "http://192.168.3.106:8080/video"
cap = None
while cap == None or not cap.isOpened():
    cap = cv2.VideoCapture(URL_CAMERA)
    if cap.isOpened():
        break
    print("Erro ao conectar com a câmera. Tentando novamente...")
    time.sleep(5)  # Espera 5 segundos antes de tentar reconectar
    
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Fila para armazenar os frames capturados
frame_queue = queue.Queue(maxsize=1)
processed_frame_queue = queue.Queue(maxsize=1)

def capture_frames():
    """Thread para capturar os frames da câmera"""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

def process_frames():
    """Thread para rodar a inferência YOLO e processar os frames"""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Reduz a resolução para acelerar a inferência
            frame_resized = cv2.resize(frame, (640, 640))

            # Colocar a imagem na GPU para rodar a inferência
            results = model(frame_resized, conf=0.90)  # Faz a inferência

            annotated_frame = results[0].plot()

            if not processed_frame_queue.full():
                processed_frame_queue.put(annotated_frame)

def generate_frames():
    """Função para enviar os frames processados via streaming"""
    while True:
        if not processed_frame_queue.empty():
            processed_frame = processed_frame_queue.get()
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Iniciar as threads de captura e processamento
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=process_frames, daemon=True).start()

@app.get("/video")
def video_feed():
    """Rota para servir o vídeo com detecção em tempo real"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
