#!/bin/bash
# ============================================================
# CodeForge - One-Click Startup Script
# Usage: bash run.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PID=""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "  ____          _      _____                    "
echo " / ___|___   __| | ___|  ___|__  _ __ __ _  ___ "
echo "| |   / _ \ / _\` |/ _ \ |_ / _ \| '__/ _\` |/ _ \\"
echo "| |__| (_) | (_| |  __/  _| (_) | | | (_| |  __/"
echo " \____\___/ \__,_|\___|_|  \___/|_|  \__, |\___|"
echo "                                      |___/      "
echo -e "${NC}"

# ---- Cleanup on exit ----
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down CodeForge...${NC}"
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null
        echo -e "  Backend (PID $BACKEND_PID) stopped."
    fi
    echo -e "${GREEN}CodeForge stopped. Goodbye!${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ---- Install dependencies ----
echo -e "${YELLOW}[1/4] Checking dependencies...${NC}"
pip install -q -r "$SCRIPT_DIR/codeforge-frontend/requirements.txt" 2>/dev/null || {
    echo -e "${RED}Failed to install dependencies. Please run:${NC}"
    echo "  pip install -r codeforge-frontend/requirements.txt"
    exit 1
}
echo -e "${GREEN}  Dependencies OK.${NC}"

# ---- Setup .env if missing ----
if [ ! -f "$SCRIPT_DIR/backend_new/.env" ]; then
    echo -e "${YELLOW}[2/4] Creating .env from template...${NC}"
    cp "$SCRIPT_DIR/backend_new/.env.example" "$SCRIPT_DIR/backend_new/.env"
    echo -e "${GREEN}  .env created. Edit backend_new/.env to add your GEMINI_API_KEY for AI mode.${NC}"
else
    echo -e "${GREEN}[2/4] .env already exists.${NC}"
fi

# ---- Start Backend ----
echo -e "${YELLOW}[3/4] Starting backend...${NC}"
cd "$SCRIPT_DIR/backend_new"
python3 jeremy_final.py &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Wait for backend to be ready (poll health endpoint)
echo -n "  Waiting for backend"
for i in $(seq 1 15); do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}  Backend running on http://localhost:8000 (PID: $BACKEND_PID)${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Check if backend actually started
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo ""
    echo -e "${RED}  Backend failed to start! Check for errors above.${NC}"
    exit 1
fi

# ---- Start Frontend ----
echo -e "${YELLOW}[4/4] Starting frontend...${NC}"
echo ""
echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}  CodeForge is ready!${NC}"
echo -e "${GREEN}  Frontend: http://localhost:8501${NC}"
echo -e "${GREEN}  Backend:  http://localhost:8000${NC}"
echo -e "${GREEN}  Login:    admin / admin123${NC}"
echo -e "${GREEN}=================================================${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop.${NC}"
echo ""

cd "$SCRIPT_DIR/codeforge-frontend"
streamlit run app.py --server.port 8501 --server.headless true
