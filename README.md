# 🛡️ EchoShield — Real-Time AI Scammer Interceptor

> Protect yourself from phone scams with live deepfake voice detection, scam tactic recognition, and an AI negotiator that wastes scammers' time.

**EchoShield** is a comprehensive multi-platform solution combining:
- 🎙️ **Real-time voice analysis** via WebSocket
- 🤖 **Deepfake detection** using PyTorch deep learning models
- 🎯 **Scam pattern recognition** with NLP-based threat classification
- 🧠 **AI negotiator** that generates strategic responses to waste scammers' time
- 📊 **Dashboard** for monitoring threats and call history
- 🔐 **End-to-end authentication** with JWT tokens

---

## 📁 Project Structure

```
├── api/                        # Vercel Serverless Function entrypoint
│   ├── index.py                # FastAPI serverless handler
│   └── requirements.txt        # Serverless-optimized Python dependencies
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── main.py             # FastAPI application entry point
│   │   ├── api/
│   │   │   ├── auth.py         # Signup, login, JWT endpoints
│   │   │   ├── audio.py        # /stream-audio, /analyze-audio
│   │   │   └── analysis.py     # /analyze, /call-logs, /stats
│   │   ├── core/
│   │   │   ├── database.py     # MongoDB Atlas connection (with mock fallback)
│   │   │   └── security.py     # JWT creation/validation, bcrypt hashing
│   │   ├── models/
│   │   │   ├── user.py         # User document model + Pydantic schemas
│   │   │   └── call_log.py     # CallLog document model + ThreatResult
│   │   └── services/
│   │       ├── transcription.py     # OpenAI Whisper speech-to-text
│   │       ├── deepfake_detector.py # PyTorch SpectrogramCNN analysis
│   │       ├── scam_detector.py     # NLP urgency pattern detection
│   │       ├── threat_classifier.py # Multi-signal threat scoring
│   │       └── negotiator.py        # AI time-wasting strategy generator
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── run.py
│   └── .env.example
├── flutter-app/                # Flutter mobile app (Android + iOS)
├── public/                     # Static assets for Vercel deployment
│   └── index.html              # High-performance web dashboard
├── web-dashboard/              # Single-file web app
│   └── index.html              # Landing page + full dashboard SPA
├── vercel.json                 # Vercel deployment & routing configuration
└── docker-compose.yml          # One-command deployment
```

---

## ⚡ Deploy to Vercel (Instant Fullstack)

EchoShield is pre-configured for seamless fullstack deployment on **Vercel** with zero complex setup.

### Option A: Via GitHub (Recommended)
1. Push this repository to **GitHub**.
2. Go to [vercel.com](https://vercel.com) and click **"Add New Project"**.
3. Import your **echoshield** repository.
4. Leave the **Build Command** and **Output Directory** empty (Vercel uses `vercel.json` automatically).
5. (Optional) In **Environment Variables**, add:
   - `MONGO_URI`: Your MongoDB Atlas connection string (e.g. `mongodb+srv://...`). If omitted, EchoShield automatically runs in high-speed in-memory mock mode.
   - `JWT_SECRET_KEY`: A 32+ character random secret.
   - `ANTHROPIC_API_KEY`: (Optional) For AI negotiator generation.
6. Click **Deploy**. Your app is live at `https://<your-project>.vercel.app`!

### Option B: Via Vercel CLI
```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Deploy from the project root
vercel --prod
```

---

## 🚀 Quick Start (Local Development)

### Prerequisites

- Python 3.11+
- Flutter 3.3+ (for mobile app)
- MongoDB Atlas account (or local MongoDB)
- Node.js (optional, for serving web dashboard)
- `ffmpeg` (required for audio processing)

---

## 🔧 Backend Setup

### 1. Install dependencies

```bash
cd echoshield/backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install AI dependencies (optional but recommended)

```bash
# Whisper for real transcription
pip install openai-whisper

# PyTorch for deepfake detection
pip install torch torchaudio librosa Pillow torchvision

# Install ffmpeg (required by Whisper)
# macOS:
brew install ffmpeg
# Ubuntu:
sudo apt install ffmpeg
# Windows: download from https://ffmpeg.org/download.html
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# MongoDB Atlas (get from https://cloud.mongodb.com)
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/echoshield

# JWT secret (change this!)
JWT_SECRET_KEY=your-super-secret-key-at-least-32-chars

# Whisper model size (tiny/base/small/medium/large)
WHISPER_MODEL=base

# Optional: Anthropic API for dynamic negotiator strategies
ANTHROPIC_API_KEY=sk-ant-...
```

> 💡 **Local MongoDB**: Use `MONGO_URI=mongodb://localhost:27017/echoshield` for a local instance.

> 💡 **Mock Mode**: If no MongoDB connection is available, the backend automatically falls back to a simulated `mock_db` mode, allowing you to test the AI and UI without setting up a database.

### 4. Run the backend

```bash
python run.py
```

Backend runs at: **http://localhost:8000**
API docs at: **http://localhost:8000/docs**

---

## 📱 Flutter Mobile App Setup

### 1. Install Flutter

Follow the official guide: https://docs.flutter.dev/get-started/install

### 2. Install dependencies

```bash
cd echoshield/flutter-app
flutter pub get
```

### 3. Configure API URL

The app defaults to `http://10.0.2.2:8000` (Android emulator → localhost).

For a real device or different server, set via `--dart-define`:

```bash
flutter run --dart-define=API_BASE_URL=http://YOUR_SERVER_IP:8000
```

### 4. Run the app

```bash
# Android (emulator or physical device)
flutter run

# Build release APK
flutter build apk --release --dart-define=API_BASE_URL=http://YOUR_SERVER:8000

# Build release AAB for Google Play Store
flutter build appbundle --release --dart-define=API_BASE_URL=http://YOUR_SERVER:8000
```

### 5. iOS Setup (macOS only)

```bash
cd ios && pod install && cd ..
flutter run -d ios
```

> ⚠️ **Microphone Permission**: The app requires microphone and phone state permissions. Accept these when prompted.

### 6. Running on Physical Device

**Android:**
```bash
# Enable USB debugging on your phone
# Connect via USB
flutter run -d all  # Shows all available devices
```

**iOS:**
```bash
# Sign the app with your Apple developer account
flutter run -d <device-id>
```

---

## 🌐 Web Dashboard Setup

### Option 1: Open directly

```bash
open echoshield/web-dashboard/index.html
```

### Option 2: Serve with Python

```bash
cd echoshield/web-dashboard
python -m http.server 3000
# Open http://localhost:3000
```

### Option 3: Serve with nginx (production)

```nginx
server {
    listen 80;
    root /path/to/echoshield/web-dashboard;
    index index.html;
    location / { try_files $uri $uri/ /index.html; }
}
```

> 💡 **CORS Note**: Make sure the backend is running and CORS is enabled for your web domain. By default, `allow_origins=["*"]` allows all origins.

---

## 🎤 Presentation & Live Demo Guide

If you are showcasing EchoShield at a presentation, follow these crucial steps to ensure the mobile app connects to your laptop's backend over the presentation network:

1. **Both devices must be on the SAME Wi-Fi network.**
2. **Find your laptop's IPv4 address:** Run `ipconfig` (Windows) or `ifconfig` (Mac/Linux). Look for an IP like `192.168.x.x` or `10.x.x.x`.
3. **Run the Flutter app pointing to that IP:**
   ```bash
   flutter run --dart-define=API_BASE_URL=http://<YOUR_LAPTOP_IP>:8000
   ```
4. **Fallback Plan:** If the school/office firewall blocks local ports, simply open the Web Dashboard (`http://localhost:8000`) and use the built-in **Analyzer** tab to manually paste scam transcripts and demonstrate the AI without the mobile app.

---

## � Running Everything Together

### Development Mode (All Components)

**Terminal 1 - Backend:**
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python run.py
# Backend runs at: http://localhost:8000
```

**Terminal 2 - Flutter App:**
```bash
cd flutter-app
flutter pub get
flutter run
# Select your target device when prompted
```

**Terminal 3 - Web Dashboard:**
```bash
cd web-dashboard
python -m http.server 3000
# Open: http://localhost:3000
```

### Using Docker Compose (Production-like)

```bash
cd echoshield
# Copy and configure env file
cp backend/.env.example backend/.env
# Edit backend/.env with your MongoDB URI

# Start everything (MongoDB, Backend, Web Dashboard)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down

# Services available at:
# Backend API:   http://localhost:8000
# Web Dashboard: http://localhost:3000
# API Docs:      http://localhost:8000/docs
# MongoDB:       localhost:27017
```

---

## ☁️ Cloud Deployment

### Backend on Railway / Render

1. Push the `backend/` folder to GitHub
2. Connect to [Railway](https://railway.app) or [Render](https://render.com)
3. Set environment variables from `.env.example`
4. Deploy — get a public URL like `https://echoshield-api.railway.app`

### Backend on AWS/GCP/Azure

```bash
# Build Docker image
docker build -t echoshield-backend ./backend

# Tag and push to ECR/GCR/ACR
docker tag echoshield-backend YOUR_REGISTRY/echoshield-backend:latest
docker push YOUR_REGISTRY/echoshield-backend:latest
```

### Web Dashboard on Vercel/Netlify

```bash
# Vercel
npx vercel ./web-dashboard

# Netlify drag-and-drop:
# Upload the web-dashboard/ folder at app.netlify.com
```

---

## 🔌 API Reference

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Health check |
| `/auth/signup` | POST | No | Create new user account |
| `/auth/login` | POST | No | Authenticate and get JWT |
| `/auth/me` | GET | Yes | Get current user info |
| `/auth/call-history` | GET | Yes | Get user's call history |
| `/stream-audio` | POST | Yes | Upload audio chunk for live analysis |
| `/analyze-audio` | POST | Yes | Analyze a complete audio recording |
| `/analyze` | POST | Yes | Analyze transcript text directly |
| `/call-logs` | GET | Yes | Get all call logs |
| `/stats` | GET | Yes | Get user statistics |

### Example: Analyze transcript

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Your account will be blocked. Send money immediately."}'
```

Response:
```json
{
  "call_log_id": "...",
  "transcript": "Your account will be blocked. Send money immediately.",
  "is_deepfake": false,
  "deepfake_confidence": 0.12,
  "urgency_detected": true,
  "urgency_score": 0.95,
  "urgency_phrases_found": ["send money immediately", "account will be blocked"],
  "overall_threat_score": 0.87,
  "threat_level": "CRITICAL",
  "negotiator_strategy": "Act confused and say you need to find your reading glasses...",
  "alert_required": true
}
```

---

## 🧠 AI Architecture

### Threat Scoring Formula

```
overall_score = (deepfake_confidence × 0.40) + (urgency_score × 0.60)
if is_deepfake: overall_score += 0.20 (capped at 1.0)

LOW      = score < 0.35
MEDIUM   = 0.35 ≤ score < 0.60
HIGH     = 0.60 ≤ score < 0.85
CRITICAL = score ≥ 0.85
```

### Deepfake Detection Pipeline

1. Load audio with `librosa` at 16kHz
2. Extract 128-band Mel spectrogram
3. Resize to 128×128 pixels
4. Pass through `SpectrogramCNN` (Conv2D → MaxPool → FC → Sigmoid)
5. Output confidence score 0.0–1.0

> 🔬 **Production**: Replace `SpectrogramCNN` with a trained model such as **RawNet2** or **AASIST** for real deepfake detection. Set `DEEPFAKE_MODEL_PATH` in `.env` to load pre-trained weights.

### Scam Detection Patterns

25+ weighted regex patterns covering:
- Urgency demands: "send money immediately", "transfer funds now"
- Account threats: "account will be blocked/suspended"
- Fear tactics: "you will be arrested", "warrant issued"
- Isolation: "don't hang up", "don't tell anyone"
- Payment manipulation: "gift card payment", "bitcoin transfer"
- Authority impersonation: IRS, Social Security, bank fraud

---

## 📋 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017/echoshield` | MongoDB connection string |
| `JWT_SECRET_KEY` | (required) | JWT signing secret — change in production! |
| `JWT_EXPIRE_MINUTES` | `1440` | Token expiry (24 hours) |
| `WHISPER_MODEL` | `base` | Whisper model: tiny/base/small/medium/large |
| `DEEPFAKE_MODEL_PATH` | `` | Path to trained deepfake model weights |
| `DEEPFAKE_THRESHOLD` | `0.6` | Score above which call is flagged as deepfake |
| `ANTHROPIC_API_KEY` | `` | For dynamic AI negotiator strategies |
| `TELEGRAM_BOT_TOKEN` | `` | Optional Telegram alert bot |
| `TELEGRAM_CHAT_ID` | `` | Optional Telegram alert recipient |

---

## 🔒 Security Notes

- Passwords are hashed with **bcrypt** (cost factor 12)
- JWTs expire after 24 hours by default
- All audio is processed server-side and not stored permanently
- Change `JWT_SECRET_KEY` before production deployment
- Use HTTPS in production (use a reverse proxy like nginx/Caddy)
- MongoDB Atlas has built-in encryption at rest

---

## 📦 Feature Checklist

- [x] User signup/login with JWT authentication
- [x] bcrypt password hashing
- [x] MongoDB Atlas storage (users, transcripts, threat scores)
- [x] OpenAI Whisper transcription (with mock fallback)
- [x] PyTorch SpectrogramCNN deepfake detection
- [x] 25+ urgency phrase pattern detection
- [x] Multi-signal threat classification (LOW/MEDIUM/HIGH/CRITICAL)
- [x] AI Negotiator strategy generator
- [x] Flutter mobile app with neon dark UI
- [x] Animated threat gauge widget
- [x] Call log history
- [x] Real-time threat alert overlay
- [x] Vibration alerts for HIGH/CRITICAL threats
- [x] Push notifications
- [x] Web dashboard with landing page
- [x] Interactive threat meter chart
- [x] AI Negotiator chatbot in web UI
- [x] Simulate scam call button for testing
- [x] Docker Compose deployment
- [x] Cloud deployment instructions

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

*Built with ❤️ to protect people from phone scammers.*
