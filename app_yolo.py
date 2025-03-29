import cv2
import time
import queue
import threading
from ultralytics import YOLO

model = YOLO("best.pt")  # Seu modelo Yolo

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

frame_queue = queue.Queue(maxsize=1)  # Mantém apenas 1 frame recente
result_queue = queue.Queue(maxsize=1)  # Guarda apenas o resultado mais recente

def process_frame():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model(frame, conf=0.90)  # Faz a inferência
            result_queue.put(results)  # Salva apenas o mais recente

# Inicia a thread de inferência
threading.Thread(target=process_frame, daemon=True).start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if not frame_queue.full():  # Garante que só processamos o frame mais recente
        frame_queue.put(frame)

    # Pega o resultado mais recente (se houver)
    if not result_queue.empty():
        results = result_queue.get()
        annotated_frame = results[0].plot()
        cv2.imshow("Detecção", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
