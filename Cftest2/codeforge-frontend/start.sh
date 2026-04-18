#!/bin/bash

# CodeForge Startup Script
# This script helps you start both the backend and frontend

echo "🔧 CodeForge Startup Script"
echo "============================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "✅ .env file created. Please update it with your configuration."
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies."
    exit 1
fi

echo "✅ Dependencies installed"
echo ""

# Ask user what to start
echo "What would you like to start?"
echo "1) Frontend only (Streamlit)"
echo "2) Backend only (FastAPI)"
echo "3) Both (in separate terminals)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting Streamlit frontend..."
        echo "📍 Opening browser at http://localhost:8501"
        echo ""
        streamlit run app.py
        ;;
    2)
        echo ""
        echo "🚀 Starting FastAPI backend..."
        echo "📍 API will be available at http://localhost:8000"
        echo ""
        cd ../backend_new 2>/dev/null || cd ../backend_new
        python3 jeremy_final.py
        ;;
    3)
        echo ""
        echo "🚀 Starting both services..."
        echo "📍 Frontend: http://localhost:8501"
        echo "📍 Backend: http://localhost:8000"
        echo ""
        echo "⚠️  Note: This requires two separate terminals."
        echo "   Terminal 1: cd backend_new && python3 jeremy_final.py"
        echo "   Terminal 2: cd codeforge-frontend && streamlit run app.py"
        echo ""
        echo "Please open two terminals and run the commands above."
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac
