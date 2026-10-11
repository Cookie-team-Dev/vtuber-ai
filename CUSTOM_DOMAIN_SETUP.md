# Custom Domain for GitHub Pages

## 1. Add CNAME File

Create `web/CNAME` with your domain:

```
kookie.yourdomain.com
```

Or for apex domain:
```
yourdomain.com
```

## 2. Configure DNS

### Subdomain (kookie.yourdomain.com)
| Type | Name | Value |
|------|------|-------|
| CNAME | kookie | `<username>.github.io` |

### Apex Domain (yourdomain.com)
| Type | Name | Value |
|------|------|-------|
| A | @ | 185.199.108.153 |
| A | @ | 185.199.109.153 |
| A | @ | 185.199.110.153 |
| A | @ | 185.199.111.153 |

## 3. Enable in GitHub

1. Repo → **Settings** → **Pages**
2. **Custom domain**: enter `kookie.yourdomain.com` (or your domain)
3. Check **"Enforce HTTPS"** (wait for DNS to propagate)

## 4. Deploy

```bash
cd /home/cookie/vtuber-ai
git add web/CNAME
git commit -m "Add custom domain"
git push origin main
```

## 5. Backend on Custom Domain

For the backend, deploy to Railway/Render/Fly.io with custom domain:

### Railway
- Project → Settings → Domains → Add `api.yourdomain.com`
- Update `web/config.js`: `window.KOOKIE_API_BASE = 'https://api.yourdomain.com'`

### Render
- Service → Settings → Custom Domains → Add `api.yourdomain.com`
- Add CNAME: `api` → `your-service.onrender.com`

### Fly.io
```bash
fly certs create api.yourdomain.com
# Add CNAME: api → your-app.fly.dev
```

## Quick Setup Script

```bash
# Run this to add CNAME
echo "kookie.yourdomain.com" > /home/cookie/vtuber-ai/web/CNAME
cd /home/cookie/vtuber-ai
git add web/CNAME
git commit -m "Add custom domain"
git push origin main
```

Replace `kookie.yourdomain.com` with your actual domain.