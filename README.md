# AI Content Detector

A free, beginner-friendly web app that estimates whether an **image**, **video**, or **text** was created by AI or a human — with a confidence percentage, a clean modern UI, dark/light mode, and local detection history.

> ⚠️ **Honesty note:** No free (or even paid) tool can detect AI content with 100% certainty. This project uses transparent, explainable statistical methods and optional open-source AI models. Treat every result as an *estimate*, not proof.

---

## 1. Project structure

```
ai-content-detector/
│
├── backend/
│   ├── app.py                    → Flask server & API routes
│   ├── requirements.txt          → Python dependencies
│   ├── .env.example              → Template for secret keys
│   └── detectors/
│       ├── __init__.py
│       ├── text_detector.py      → Text analysis (statistics + optional HF model)
│       ├── image_detector.py     → Image analysis (pixel stats + optional HF model)
│       └── video_detector.py     → Video analysis (frame sampling)
│
├── frontend/
│   ├── index.html                → Home page
│   ├── detector.html             → Detector page (image/video/text tabs)
│   ├── about.html                → About & limitations page
│   ├── contact.html              → Contact page
│   ├── css/style.css             → All styling
│   ├── js/
│   │   ├── main.js               → Dark mode, mobile nav, FAQ accordion
│   │   ├── detector.js           → Upload, drag-and-drop, API calls, results
│   │   └── history.js            → LocalStorage detection history
│   └── assets/                   → (place any extra images/icons here)
│
└── README.md                     → You are here
```

---

## 2. How the detection actually works (plain English)

### Text
We count sentences and words, then measure:
- **Burstiness** — humans mix short and long sentences; AI text is often more uniform.
- **Vocabulary variety** — ratio of unique words to total words.
- **Repetition** — how often the same few words are reused.

These three numbers combine into an "AI probability" score from 5% to 95% (we never claim 100%).

### Images
We resize the image and measure:
- **Noise level** — real photos have natural sensor noise; AI images are often too smooth.
- **Color spread** — how varied the colors are.
- **Edge energy** — how much fine detail/texture exists.

### Videos
We can't analyze a whole video efficiently with free tools, so we:
1. Pull 6 evenly-spaced frames out of the video (like taking screenshots).
2. Run each frame through the **same** image analysis above.
3. Average the results into one final verdict.

### Real AI model detection (required)
This app uses real, free, open-source AI models via Hugging Face's **current** Inference Providers router (`https://router.huggingface.co`) — the old `api-inference.huggingface.co` endpoint has been permanently shut down by Hugging Face and will not work.

- Text: [`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01) — currently leads the independent RAID Benchmark for AI-text detection, MIT-licensed, actively maintained.
- Image: [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector) — actively maintained, used in 100+ community Spaces.
- Video: samples 4 frames and runs each through the image model above.

All the actual networking, retry, and error-handling logic lives in one shared file: `backend/detectors/base.py`. If you ever want to swap providers (e.g. use a paid API instead), you only need to change that one file — the rest of the app doesn't need to change.

You need a **free Hugging Face account and API token** for this to work — see Step 2 below. If the token is missing or invalid, the app returns a clear "Setup needed" message instead of crashing. If a model is temporarily "waking up" or Hugging Face's free tier rate-limits you, the app retries automatically and shows an honest error if it still can't get through. Every request is also logged (see your terminal locally, or Render's "Logs" tab in production) to make problems easy to diagnose.

---

## 3. Running it on your own computer

### Step 1 — Install Python
Download Python 3.10+ from [python.org](https://www.python.org/downloads/) if you don't already have it.

### Step 2 — Set up the backend
Open a terminal (Command Prompt / Terminal app) and run:

```bash
cd ai-content-detector/backend
pip install -r requirements.txt
```

**Required:** Copy `.env.example` to `.env`, then get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (click "New token", choose the "Read" role) and paste it in:
```
HUGGINGFACE_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Start the server:

```bash
python app.py
```

You should see something like:
```
* Running on http://127.0.0.1:5000
```
Leave this terminal window open — this is your backend running.

### Step 3 — Open the frontend
You don't need to "install" anything for the frontend. Just open `frontend/index.html` in your browser — or, for the best results (so drag-and-drop and fetch requests behave correctly), right-click `index.html` in VS Code and choose **"Open with Live Server"** (a free VS Code extension), which serves it at something like `http://127.0.0.1:5500`.

Make sure `frontend/js/detector.js` has:
```js
const BACKEND_URL = 'http://127.0.0.1:5000';
```
This must match wherever your Flask server is running.

### Step 4 — Try it
Go to the **Detector** page, upload an image/video or paste some text, and press **Analyze**.

---

## 4. Deploying for free (GitHub + Render)

### Step 1 — Push your project to GitHub
1. Create a free account at [github.com](https://github.com) if you don't have one.
2. Create a new repository (e.g. `ai-content-detector`).
3. In your project folder, run:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/ai-content-detector.git
   git push -u origin main
   ```

Create a `.gitignore` file in the project root with this content so you never upload secrets:
```
backend/.env
__pycache__/
*.pyc
```

### Step 2 — Deploy the backend on Render
1. Go to [render.com](https://render.com) and sign up (free).
2. Click **New +** → **Web Service**.
3. Connect your GitHub repo.
4. Set:
   - **Root directory:** `backend`
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app` (see note below)
   - **Environment variable (optional):** `HUGGINGFACE_API_TOKEN` = your token
5. Click **Create Web Service**. Render will give you a live URL like:
   `https://ai-content-detector-backend.onrender.com`

> **Note:** Flask's built-in server (`python app.py`) is fine for testing but not for production. Add `gunicorn==22.0.0` to `requirements.txt` and use it as your Render start command, as shown above.

### Step 3 — Update the frontend to point to your live backend
In `frontend/js/detector.js`, change:
```js
const BACKEND_URL = 'http://127.0.0.1:5000';
```
to your Render URL:
```js
const BACKEND_URL = 'https://ai-content-detector-backend.onrender.com';
```

### Step 4 — Deploy the frontend (two free options)
**Option A — Render Static Site:**
1. New + → **Static Site**.
2. Root directory: `frontend`
3. Build command: (leave blank)
4. Publish directory: `.`

**Option B — GitHub Pages:**
1. In your GitHub repo settings → **Pages**.
2. Set source to the `frontend` folder on the `main` branch.
3. GitHub will give you a live link like `https://YOUR-USERNAME.github.io/ai-content-detector/`.

That's it — your site is now live and free!

---

## 5. Notes on free-tier limits
- Render's free web services "sleep" after inactivity and take ~30–50 seconds to wake up on the next request — this is normal on the free tier.
- Hugging Face's free Inference API has rate limits; if you rely on it heavily, requests may occasionally fail (the app will automatically fall back to the offline method).

---

## 6. What you could add next
- User accounts + server-side history (instead of browser-only history)
- A queue system for large video files
- A stronger, fine-tuned open-source model hosted on Hugging Face Spaces
