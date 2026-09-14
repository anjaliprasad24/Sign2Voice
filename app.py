"""
Flask-SocketIO backend server.
Owns: local server deployment, real-time inference tunneling, and
state management (per Teammate 2 - Backend & Systems Integration Lead).

WebSocket Contract (agreed with Teammate 3 - Frontend):

    Client -> Server   event: "video_frame"
        payload: { "image": "<base64-encoded JPEG from <canvas>>" }

    Server -> Client   event: "prediction"
        payload: { "text": "Thank You", "confidence": 0.89 }

    Server -> Client   event: "sentence_update"
        payload: { "sentence": "Hello Thank You " }

    Client -> Server   event: "reset_sentence"
        payload: (none)

Run with:
    python app.py
Then open:
    http://localhost:5000
"""
from flask import Flask, render_template, request
from flask_socketio import SocketIO

import config
from utils.image_processing import apply_clahe
from utils.landmark_extraction import decode_base64_frame, extract_landmarks
from utils.inference_engine import SequenceBuffer, extract_features, predict, USE_MOCK_MODEL
from utils.state_machine import GestureStateMachine

app = Flask(__name__)
app.config["SECRET_KEY"] = "sign-language-dev-secret"  # replace before any real deployment
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Per-client session state, keyed by Socket.IO session id (sid). Each
# connected browser tab gets its own rolling frame buffer and state
# machine so multiple users never bleed into each other's sentences.
_sessions = {}


def _get_session(sid):
    if sid not in _sessions:
        _sessions[sid] = {
            "buffer": SequenceBuffer(),
            "state_machine": GestureStateMachine(),
        }
    return _sessions[sid]


@app.route("/")
def index():
    """Serves Teammate 3's frontend dashboard (templates/index.html)."""
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    _get_session(request.sid)
    print(f"[connect] client {request.sid} connected"
          f"{' (MOCK MODEL - model files not found yet)' if USE_MOCK_MODEL else ''}")


@socketio.on("disconnect")
def handle_disconnect():
    _sessions.pop(request.sid, None)
    print(f"[disconnect] client {request.sid} disconnected")


@socketio.on("video_frame")
def handle_video_frame(payload):
    """
    Main real-time inference loop, run once per incoming frame:

        decode -> CLAHE -> MediaPipe landmarks -> feature extraction ->
        sequence buffer -> LSTM prediction -> debounce state machine ->
        emit confirmed results only.
    """
    session = _get_session(request.sid)

    frame = decode_base64_frame(payload.get("image", ""))
    if frame is None:
        return

    enhanced_frame = apply_clahe(frame)
    hands_data = extract_landmarks(enhanced_frame)

    # Even with no hand detected we push a (zero) feature vector so the
    # temporal window keeps advancing - a dropped hand is itself signal
    # the state machine uses to recognize the NEUTRAL/REST gap.
    features = extract_features(hands_data)
    session["buffer"].push(features)

    if not session["buffer"].is_ready():
        return

    label, confidence = predict(session["buffer"].as_array())
    result = session["state_machine"].update(label, confidence)

    if result["event"] == "word_committed":
        socketio.emit(
            "prediction",
            {"text": result["label"], "confidence": result["confidence"]},
            to=request.sid,
        )
        socketio.emit(
            "sentence_update", {"sentence": result["sentence"]}, to=request.sid
        )
    elif result["event"] == "space_inserted":
        socketio.emit(
            "sentence_update", {"sentence": result["sentence"]}, to=request.sid
        )


@socketio.on("reset_sentence")
def handle_reset(_payload=None):
    session = _get_session(request.sid)
    session["state_machine"].reset_sentence()
    socketio.emit("sentence_update", {"sentence": ""}, to=request.sid)


if __name__ == "__main__":
    socketio.run(app, host=config.HOST, port=config.PORT, debug=config.DEBUG)
