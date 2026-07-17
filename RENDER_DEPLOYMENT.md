# 🚀 Render Deployment Guide

Complete step-by-step guide to deploy EAAS Prototype on Render.

## Prerequisites

- GitHub account with the EAAS repository pushed
- Render account (free tier available)
- Git installed locally

## Step 1: Push Code to GitHub

```bash
# Navigate to project directory
cd EAAS_Software_Prototype

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: EAAS Prototype"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
git push -u origin main
```

## Step 2: Create Render Account

1. Go to [render.com](https://render.com)
2. Click "Sign up"
3. Sign up with GitHub (recommended for easy authorization)
4. Authorize Render to access your GitHub repositories

## Step 3: Create Web Service on Render

1. **Dashboard**: Click "New +" button in top right
2. **Select "Web Service"**
3. **Connect Repository**:
   - Click "Connect account" if needed
   - Search for your `EAAS_Software_Prototype` repository
   - Click "Connect"

4. **Configure Service**:
   - **Name**: `eaas-prototype` (or your preferred name)
   - **Region**: Select closest to you (e.g., US, EU)
   - **Branch**: `main`
   - **Runtime**: `Python 3`

5. **Build & Start Commands**:
   - **Build Command**:
     ```
     pip install -r requirements.txt && python -m eaas.db
     ```
   - **Start Command**:
     ```
     gunicorn -w 4 -b 0.0.0.0:$PORT eaas.app:app
     ```

6. **Plan**: Select "Free" (sufficient for testing)

7. **Environment Variables** (if needed):
   - Leave empty for now - defaults are configured

8. **Advanced Settings** (optional):
   - Auto-deploy: Enabled (automatically redeploy on GitHub push)

9. Click **"Create Web Service"**

## Step 4: Monitor Deployment

1. After clicking "Create Web Service", Render will:
   - Clone your GitHub repository
   - Install dependencies from `requirements.txt`
   - Initialize the database
   - Start the application

2. You'll see build logs in real-time. Wait for:
   ```
   Deployment successful!
   Live URL: https://<service-name>.onrender.com
   ```

3. Your app will be available at the Live URL

## Step 5: Test Your Application

1. Open the Live URL in your browser: `https://<service-name>.onrender.com`
2. You should see the EAAS home page
3. Test functionality:
   - Register a new user
   - Perform a login scan
   - View admin logs

## Step 6: Use Custom Domain (Optional)

1. Go to Service Settings
2. Scroll to "Custom Domain"
3. Add your domain (requires DNS configuration)

## Troubleshooting

### Build Fails

**Error**: `Collecting opencv-contrib-python`
- **Solution**: This is normal on first deploy. The build includes compilation. Be patient (can take 5-10 minutes)

**Error**: `ModuleNotFoundError: No module named 'cv2'`
- **Solution**: Ensure `opencv-contrib-python-headless` (not regular opencv) is in requirements.txt

### Application Crashes After Deployment

Check logs:
1. Go to Service Dashboard
2. Click "Logs" tab
3. Look for error messages
4. Common issues:
   - Missing environment variables
   - Database initialization failed
   - Port not set correctly

### Database Issues

If database doesn't initialize:
1. Open service shell (in Render dashboard)
2. Run: `python -m eaas.db`
3. Restart service

### Camera/Face Detection Not Working

This is expected on Render (headless server):
- Face capture requires a client-side browser with webcam
- Render serves the web interface, client browser provides camera
- This is normal behavior

## Updating Your Application

After making changes locally:

```bash
# Commit and push to GitHub
git add .
git commit -m "Update: [description of changes]"
git push origin main
```

Render will automatically detect the push and redeploy if auto-deploy is enabled.

## Performance Tips

1. **Free Plan Limitations**:
   - Service spins down after 15 minutes of inactivity
   - First request after spin-down takes ~30 seconds
   - Adequate for testing/demo purposes

2. **To Upgrade**:
   - Go to Service Settings
   - Change plan to "Starter" or "Standard"
   - Recommended for production use

3. **Optimize**:
   - ML models are cached after first load
   - Database is persistent (not lost between restarts)
   - Consider reducing image capture resolution if needed

## Security Notes

1. **Database**: SQLite is suitable for development/testing but not production at scale
2. **HTTPS**: Render provides automatic SSL/TLS
3. **Environment Variables**: For production, add sensitive config via Render dashboard
4. **Rate Limiting**: Consider adding rate limiting for login attempts (currently not implemented)

## Monitoring

### View Logs

1. Dashboard → Service → Logs tab
2. Real-time application logs visible here

### CPU & Memory

1. Dashboard → Service → Metrics tab
2. View usage statistics

### Access Logs

1. Visit `/admin/logs` in your deployed application
2. View all authentication attempts
3. Database persists between restarts

## Scaling

When ready for production:

1. Upgrade from Free → Starter/Standard plan
2. Add more workers: Modify start command to `gunicorn -w 8 -b 0.0.0.0:$PORT eaas.app:app`
3. Consider PostgreSQL instead of SQLite
4. Enable HTTPS (automatic on Render)
5. Set up monitoring/alerting

## Need Help?

- **Render Docs**: https://render.com/docs
- **Flask Deployment**: https://flask.palletsprojects.com/deployment/
- **GitHub Issues**: Open an issue in the repository

---

**Last Updated**: 2024
