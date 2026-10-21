# Kookie Web - GitHub Pages Deployment

## Quick Deploy

1. **Push to GitHub** - This repo deploys the `web/` folder to GitHub Pages automatically
2. **Enable Pages** - In repo Settings → Pages → Source: "GitHub Actions"
3. **Get URL** - Your site will be at `https://<username>.github.io/<repo-name>/`

## Backend Required

The frontend needs a Python backend (`webui.py`) for:
- API key relay (`/api/provider`)
- Family authentication (`/api/whoami`, `/api/chat`)
- Memory persistence (`/api/memory`, `/api/remember`)
- TTS (`/api/tts`)

## Backend Hosting Options

### Option 1: Railway (Recommended)
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Add these environment variables in Railway dashboard:
- `KOOKIE_WEB_FAMILY_KEY=__SET_IN_ENV__`
- `KOOKIE_WEB_PORT=8080`

### Option 2: Render
```yaml
# render.yaml
services:
  - type: web
    name: kookie-web
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: python webui.py
    envVars:
      - key: KOOKIE_WEB_FAMILY_KEY
        value: __SET_IN_ENV__
      - key: KOOKIE_WEB_PORT
        value: 8080
```

### Option 3: Fly.io
```bash
fly launch --no-deploy
fly deploy
```

### Option 4: VPS (DigitalOcean, Linode, etc.)
```bash
# On your server
git clone <your-repo>
cd vtuber-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
KOOKIE_WEB_FAMILY_KEY=__SET_IN_ENV__ KOOKIE_WEB_PORT=8080 nohup python webui.py &
```

## Connect Frontend to Backend

After deploying backend, update the frontend API base URL in `web/index.html`:

```javascript
// Find this line (around line 1054):
var res = await fetch('/api/provider', {

// Change to your backend URL:
var res = await fetch('https://your-backend.railway.app/api/provider', {
```

Or set via environment variable at build time (create `web/config.js`):
```javascript
window.KOOKIE_API_BASE = 'https://your-backend.railway.app';
```

Then update all fetch calls:
```javascript
fetch(window.KOOKIE_API_BASE + '/api/provider', ...
```

## Local Development

```bash
# Terminal 1: Start backend
KOOKIE_WEB_PORT=8080 KOOKIE_WEB_FAMILY_KEY="__SET_IN_ENV__" python webui.py

# Terminal 2: Serve frontend (optional)
cd web && python -m http.server 3000
# Visit http://localhost:3000
```

## Family Keys

- **Family**: `__SET_IN_ENV__` + Discord ID
- **Stranger**: `kookieAIdiscord.me`

## Developer Access

Add your Discord ID hash to `web/index.html`:
```javascript
var DEV_ID_HASHES = ['your-sha256-discord-id-hash'];
```

Generate hash:
```bash
python -c "import hashlib; print(hashlib.sha256(b'1373614699890868248').hexdigest())"
```