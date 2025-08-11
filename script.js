const video = document.getElementById('video-feed');
const canvas = document.getElementById('detection-canvas');
const ctx = canvas.getContext('2d');

let session;
let modelLoaded = false;

// โหลดโมเดล YOLOv8 (ONNX)
async function loadONNXModel() {
    session = new onnx.InferenceSession({ backendHint: 'webgl' });
    await session.loadModel('./yolov8n.onnx');
    modelLoaded = true;
    console.log("✅ ONNX model loaded");
}

// เปิดกล้อง
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
    video.srcObject = stream;
    await new Promise(resolve => {
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve();
        };
    });
    video.play();
}

// โหลดวิดีโอจากไฟล์
async function setupVideoFile(file) {
    video.srcObject = null;
    video.src = URL.createObjectURL(file);
    await new Promise(resolve => {
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            resolve();
        };
    });
    video.play();
}

// ประมวลผลเฟรม
async function runDetection() {
    if (!modelLoaded) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // ✅ ดึง pixel จาก video frame
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = preprocessImage(imageData, 640, 640); // ปรับตาม input size โมเดล

    const outputMap = await session.run({ images: inputTensor });
    const output = outputMap[session.outputNames[0]];

    drawDetections(output.data);

    requestAnimationFrame(runDetection);
}

// แปลงภาพให้เข้ากับ YOLOv8
function preprocessImage(imageData, modelWidth, modelHeight) {
    const data = imageData.data;
    const resized = new Float32Array(modelWidth * modelHeight * 3);

    // resize + normalize (0-1) + swap RGB
    let offset = 0;
    for (let i = 0; i < data.length; i += 4) {
        resized[offset] = data[i] / 255;     // R
        resized[offset + modelWidth * modelHeight] = data[i + 1] / 255; // G
        resized[offset + 2 * modelWidth * modelHeight] = data[i + 2] / 255; // B
        offset++;
    }

    return new onnx.Tensor(resized, 'float32', [1, 3, modelWidth, modelHeight]);
}

// วาด Bounding Box
function drawDetections(predictions) {
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.font = "16px Arial";
    ctx.fillStyle = "red";

    // TODO: แปลง predictions → bounding box ตาม format YOLOv8
    // ตัวอย่างด้านล่างเป็น placeholder สมมติ
    // for (let det of detections) { ctx.strokeRect(...); ctx.fillText(...); }
}

// เริ่มใช้งาน
document.getElementById('start-camera').addEventListener('click', async () => {
    await loadONNXModel();
    await setupCamera();
    runDetection();
});

document.getElementById('upload-video').addEventListener('change', async (e) => {
    if (e.target.files.length > 0) {
        await loadONNXModel();
        await setupVideoFile(e.target.files[0]);
        runDetection();
    }
});
