const socket = io("http://localhost:5000"); // Conecta ao servidor WebSocket
console.log(socket);

const video = document.getElementById("video-stream");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

socket.on("frame", (data) => {
    // Atualiza a imagem do stream
    video.src = "data:image/jpeg;base64," + data.image;

    // Desenha as bounding boxes
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    data.detections.forEach(box => {
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
    });
});
