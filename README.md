# 🤖 ActivityVision — Human Activity Detection

Real-time human activity detection using **MediaPipe Pose** + **rule-based classification**, with a polished dark dashboard frontend.

---

## 📁 Structure

```
human_activity_detection/
├── backend/
│   ├── app.py            ← Flask REST API
│   └── requirements.txt
└── frontend/
    └── index.html        ← Single-file dashboard
```

---

## 🚀 Quick Start

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

### Frontend
```bash
cd frontend
python -m http.server 8080
# → http://localhost:8080
```
> **Note:** Open as HTTP (not file://) so webcam access works in Chrome.

---

## 🎯 Detected Activities

| Activity | Emoji | How it's detected |
|---|---|---|
| Standing | 🧍 | Default upright pose |
| Sitting | 🪑 | Knee angle < 120° + hip above knee |
| Walking | 🚶 | Moderate hip vertical asymmetry |
| Running | 🏃 | Strong hip vertical asymmetry |
| Squatting | 🏋️ | Knee angle < 115° + low hip |
| Lying Down | 🛌 | Torso inclined > 55° from vertical |
| Arms Raised | 🙌 | Wrists above shoulder level |
| Push-up Position | 💪 | Inclined torso + extended legs + bent elbows |

---

## 🔌 API Reference

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check + uptime |
| POST | `/detect` | Detect activity in a frame |
| GET | `/history` | Session history |
| POST | `/reset` | Reset session |

### POST `/detect`
**Request:**
```json
{ "image": "<base64 JPEG>" }
```
**Response:**
```json
{
  "detected": true,
  "activity": "Walking",
  "confidence": 0.76,
  "emoji": "🚶",
  "color": "#06d6a0",
  "annotated_image": "data:image/jpeg;base64,...",
  "landmarks": [{"x":0.5,"y":0.3,"z":-0.1,"v":0.99}, ...],
  "history": [...],
  "frame": 137
}
```

---

## 🛠 Tech Stack

| Layer | Tech |
|---|---|
| Frontend | HTML5 / CSS3 / Vanilla JS |
| Backend | Python 3.10+ / Flask |
| Pose Detection | MediaPipe Pose (33 landmarks) |
| Computer Vision | OpenCV |
| Communication | REST / JSON / base64 frames |

---

## 💡 Extending

- **ML upgrade:** Replace rule-based classifier with a trained LSTM/Random Forest on keypoint sequences
- **More activities:** Add yoga poses, jumping jacks, bicep curls via angle rules
- **WebSocket:** Replace polling interval with WebSocket for lower latency
- **Video file support:** Add `/detect_video` endpoint for offline MP4 analysis
- **Recording:** Save session as CSV/JSON for later analysis