import cv2
import threading
import queue
from ultralytics import YOLO

cap = cv2.VideoCapture("url")
model = YOLO("best.pt")  # Seu modelo YOLOv8

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
