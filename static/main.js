document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("capture-canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const predictedWordEl = document.getElementById("predicted-word");
  const confidenceValueEl = document.getElementById("confidence-value");
  const confidenceFillEl = document.getElementById("confidence-fill");
  const sentenceBufferEl = document.getElementById("sentence-buffer");
  const lightWarningEl = document.getElementById("light-warning");
  const connectionStatusEl = document.getElementById("connection-status");
  const fpsCounterEl = document.getElementById("fps-counter");
  const clearBtn = document.getElementById("clear-btn");

  // Constants & Config
  const TARGET_FPS = 30;
  const FRAME_INTERVAL = 1000 / TARGET_FPS;
  const LUX_THRESHOLD = 50; // Average pixel brightness threshold
  let lastFrameTime = 0;
  let frameCount = 0;
  let lastFpsUpdate = Date.now();

  // 1. Initialize WebSocket Connection
  const socket = io();

  socket.on("connect", () => {
    connectionStatusEl.textContent = "Connected";
    connectionStatusEl.style.color = "#4ade80";
  });

  socket.on("disconnect", () => {
    connectionStatusEl.textContent = "Disconnected";
    connectionStatusEl.style.color = "#f87171";
  });

  // 2. Ingest Predictions & Trigger Web Speech API Contract
  socket.on("gesture_prediction", (data) => {
    // Contract format: { text: "...", confidence: 0.89 }
    const { text, confidence } = data;

    // Update UI elements
    predictedWordEl.textContent = text;
    const confidencePct = Math.round(confidence * 100);
    confidenceValueEl.textContent = `${confidencePct}%`;
    confidenceFillEl.style.width = `${confidencePct}%`;

    if (text && text !== "NEUTRAL" && text !== "") {
      // Append word to sentence buffer
      sentenceBufferEl.textContent = sentenceBufferEl.textContent
        ? `${sentenceBufferEl.textContent} ${text}`
        : text;

      // Synthesize Speech Asynchronously
      speakWord(text);
    }
  });

  function speakWord(text) {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel(); // Drop any lagging queue
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  }

  // 3. Setup Webcam Feed
  async function initWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false
      });
      video.srcObject = stream;
    } catch (err) {
      console.error("Webcam access error:", err);
      connectionStatusEl.textContent = "Camera Error";
      connectionStatusEl.style.color = "#f87171";
    }
  }

  // 4. Ambient Light Guard (Average Brightness)
  function checkAmbientLighting(imageData) {
    const data = imageData.data;
    let totalBrightness = 0;
    const totalPixels = data.length / 4;

    // Sample step to minimize overhead
    const step = 8;
    let sampleCount = 0;

    for (let i = 0; i < data.length; i += 4 * step) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      // Standard luminance formula
      const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
      totalBrightness += luminance;
      sampleCount++;
    }

    const avgBrightness = totalBrightness / sampleCount;

    if (avgBrightness < LUX_THRESHOLD) {
      lightWarningEl.classList.remove("hidden");
    } else {
      lightWarningEl.classList.add("hidden");
    }
  }

  // 5. Frame Capture, Downscale (480x360), and Socket Emission Loop
  function streamLoop(timestamp) {
    requestAnimationFrame(streamLoop);

    const elapsed = timestamp - lastFrameTime;
    if (elapsed < FRAME_INTERVAL) return;

    lastFrameTime = timestamp - (elapsed % FRAME_INTERVAL);

    if (video.readyState === video.HAVE_ENOUGH_DATA) {
      // Draw to hidden downscaling canvas (480x360)
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Extract pixel data for Lux check
      const frameData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      checkAmbientLighting(frameData);

      // Convert to compressed JPEG data URL and emit over WebSocket
      const framePayload = canvas.toDataURL("image/jpeg", 0.6);
      socket.emit("video_frame", { image: framePayload });

      // Calculate runtime client FPS
      frameCount++;
      const now = Date.now();
      if (now - lastFpsUpdate >= 1000) {
        fpsCounterEl.textContent = frameCount;
        frameCount = 0;
        lastFpsUpdate = now;
      }
    }
  }

  clearBtn.addEventListener("click", () => {
    sentenceBufferEl.textContent = "";
  });

  // Start processes
  initWebcam().then(() => {
    requestAnimationFrame(streamLoop);
  });
});