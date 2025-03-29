import cv2
import time
import torch
from fastapi import FastAPI
from ultralytics import YOLO
from fastapi.responses import StreamingResponse

# Inicializa a API
app = FastAPI()

# Carrega o modelo YOLO com suporte à GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt").to(device)

# Conecta à câmera do celular (troque o IP conforme necessário)
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

def generate_frames():
    while True:
        for _ in range(1):  # Descartamos 3 frames para reduzir o delay
            cap.grab()

        success, frame = cap.read()
        if not success:
            break  # Se falhar na captura, sai do loop
        
        # Redimensiona o frame para o tamanho de imagem que o modelo foi treinado
        frame = cv2.resize(frame, (640, 640))

        # Converte para RGB (YOLO espera isso)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Faz a inferência com YOLO
        results = model(frame, conf=0.90)

        # Obtém a imagem com as bounding boxes desenhadas
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Converte para formato JPEG
        _, jpeg = cv2.imencode(".jpg", annotated_frame)

        # Retorna os frames no formato de streaming MJPEG
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               bytearray(jpeg) + b"\r\n")

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Roda a API automaticamente quando executar este script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
