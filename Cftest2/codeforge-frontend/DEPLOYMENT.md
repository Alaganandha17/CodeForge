# 🚀 CodeForge Deployment Guide

Complete guide for setting up and deploying CodeForge

## 📋 Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Backend Configuration](#backend-configuration)
3. [Frontend Configuration](#frontend-configuration)
4. [Testing](#testing)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting](#troubleshooting)

## 🏠 Local Development Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/jeremyyyS/CodeForge.git
cd CodeForge
```

### Step 2: Set Up Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt

# Create .env file for backend
cat > .env << EOF
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-2.5-flash
EOF

# Test backend
python jeremy_final.py
```

Backend should now be running on `http://localhost:8000`

### Step 3: Set Up Frontend

```bash
# Open a new terminal
cd CodeForge/frontend

# Create virtual environment (or use same as backend)
python3 -m venv venv
source venv/bin/activate

# Install frontend dependencies
pip install -r requirements.txt

# Create .env file for frontend
cp .env.example .env

# Edit .env if needed
nano .env  # or your preferred editor

# Start frontend
streamlit run Home.py
```

Frontend should now be running on `http://localhost:8501`

### Step 4: Access the Application

1. Open browser to `http://localhost:8501`
2. Login with default credentials:
   - Admin: `admin` / `admin123`
   - User: `user` / `user123`

## ⚙️ Backend Configuration

### Environment Variables

Create `.env` in backend directory:

```env
# Required
GEMINI_API_KEY=your_api_key_here
MODEL_NAME=gemini-2.5-flash

# Optional
API_TIMEOUT=30
MAX_CODE_LENGTH=10000
BENCHMARK_RUNS=3
BENCHMARK_ITERATIONS=50
```

### Getting Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or sign in to your Google account
3. Click "Create API Key"
4. Copy the key and paste in `.env`

### Backend Files Structure

```
backend/
├── jeremy_final.py          # Main FastAPI app
├── ai_explainer.py          # AI explanation generator
├── config.py                # Configuration
├── rules_engine.py          # Rule-based optimizer
├── rule_transformer.py      # Code transformations
├── llm_optimizer.py         # Gemini integration
├── semantic_search.py       # Semantic pattern detection
├── safety.py                # Safety validation
├── metrics.py               # Confidence & explainability
├── utils.py                 # Utility functions
└── .env                     # Environment variables
```

## 🎨 Frontend Configuration

### Environment Variables

Create `.env` in frontend directory:

```env
# Backend URL
BACKEND_URL=http://localhost:8000

# User credentials (JSON array)
APP_USERS_JSON=[{"u":"admin","p":"admin123","role":"admin"},{"u":"user","p":"user123","role":"user"}]
```

### Adding Custom Users

Edit the `APP_USERS_JSON` in `.env`:

```env
APP_USERS_JSON=[
  {"u":"admin","p":"secure_admin_password","role":"admin"},
  {"u":"john","p":"john_password","role":"user"},
  {"u":"jane","p":"jane_password","role":"user"}
]
```

### Frontend Files Structure

```
frontend/
├── Home.py                  # Login page (entry point)
├── pages/
│   ├── 1_Dashboard.py       # Main optimization UI
│   ├── 2_History.py         # History page
│   ├── 3_Api_Docs.py        # API documentation
│   ├── 4_About.py           # About page
│   ├── 5_Upload_Code.py     # File upload
│   ├── 6_Dataset_Tools.py   # Dataset management
│   └── 7_Admin_Login.py     # Admin panel
├── utils/
│   ├── __init__.py
│   ├── api.py               # API client
│   └── auth.py              # Authentication
├── assets/
│   └── examples/
│       └── fib_unoptimized.py
├── requirements.txt
├── .env
└── README.md
```

## 🧪 Testing

### Test Backend

```bash
cd backend

# Test health endpoint
curl http://localhost:8000/

# Test optimization
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{"code":"def sum_list(lst):\n    total = 0\n    for i in lst:\n        total += i\n    return total"}'
```

### Test Frontend

1. Login with test credentials
2. Try optimizing the default example code
3. Upload a Python file
4. Check History page
5. View API Docs
6. Admin users: Test Dataset Tools and Admin panel

## 🌐 Production Deployment

### Option 1: Docker Deployment

```bash
# Backend Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/ .
RUN pip install -r requirements.txt
ENV GEMINI_API_KEY=${GEMINI_API_KEY}
EXPOSE 8000
CMD ["python", "jeremy_final.py"]

# Frontend Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY frontend/ .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Option 2: Cloud Deployment (e.g., Heroku, AWS, GCP)

**Backend (Heroku):**

```bash
# In backend directory
echo "web: python jeremy_final.py" > Procfile
git init
heroku create codeforge-backend
heroku config:set GEMINI_API_KEY=your_key_here
git add .
git commit -m "Deploy backend"
git push heroku master
```

**Frontend (Streamlit Cloud):**

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `frontend/Home.py` as main file
5. Add secrets in Streamlit Cloud dashboard:
   - `BACKEND_URL=https://your-backend-url.herokuapp.com`
   - `APP_USERS_JSON=...`

### Option 3: VPS Deployment (Ubuntu)

```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv nginx

# Clone repository
git clone https://github.com/jeremyyyS/CodeForge.git
cd CodeForge

# Backend setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Create systemd service for backend

# Frontend setup
cd ../frontend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Create systemd service for frontend

# Configure Nginx as reverse proxy
```

## 🔧 Troubleshooting

### Backend Issues

**Issue**: `GEMINI_API_KEY not found`

**Solution**:
```bash
# Make sure .env exists in backend directory
cd backend
cat .env  # Should show GEMINI_API_KEY=...
```

**Issue**: `Module not found` errors

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Frontend Issues

**Issue**: `Connection refused` to backend

**Solution**:
```bash
# Check if backend is running
curl http://localhost:8000/
# If not, start it: python jeremy_final.py
```

**Issue**: Login not working

**Solution**:
```bash
# Check .env file in frontend directory
# Verify APP_USERS_JSON is valid JSON
# Try default credentials: admin/admin123
```

### Performance Issues

**Issue**: Optimization timing out

**Solution**:
- Reduce code complexity
- Use Rules-Only mode instead of AI
- Increase `API_TIMEOUT` in backend config.py

**Issue**: Slow response times

**Solution**:
- Check backend server resources (CPU, RAM)
- Consider caching frequent optimizations
- Use CDN for frontend assets

## 📊 Monitoring

### Backend Logs

```bash
# View backend logs
tail -f backend.log

# Or if running in terminal
python jeremy_final.py  # Logs will appear in console
```

### Frontend Logs

```bash
# Streamlit shows logs in terminal
streamlit run Home.py  # Logs appear here
```

### Health Checks

```bash
# Backend health
curl http://localhost:8000/

# Should return:
# {"message":"SafeOpt Code Optimizer","status":"running","version":"1.0"}
```

## 🔐 Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Passwords**: Use strong passwords in production
3. **HTTPS**: Always use HTTPS in production
4. **Rate Limiting**: Implement rate limiting on API endpoints
5. **Input Validation**: Already implemented in backend
6. **Session Management**: Consider adding JWT tokens for production

## 📈 Scaling

For high traffic:

1. **Load Balancing**: Use Nginx or cloud load balancer
2. **Database**: Migrate from session storage to PostgreSQL/MongoDB
3. **Caching**: Implement Redis for optimization results
4. **CDN**: Use CloudFlare or similar for static assets
5. **Horizontal Scaling**: Deploy multiple backend instances

## 🎯 Next Steps

1. Set up monitoring (Sentry, New Relic)
2. Implement database for persistent storage
3. Add automated testing (pytest)
4. Set up CI/CD pipeline (GitHub Actions)
5. Create backup and recovery procedures

---

**Need Help?**
- Check the main README.md
- Review API documentation in the app
- Contact via GitHub issues
