from flask import Flask
from flask_socketio import SocketIO, emit
from detect import get_frame  # Importa o módulo de detecção

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on("connect")
def handle_connect():
    print("Cliente conectado")
    def send_frames():
        for frame_data in get_frame():
            socketio.emit("frame", frame_data)  # Envia os dados via WebSocket
    socketio.start_background_task(send_frames)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
