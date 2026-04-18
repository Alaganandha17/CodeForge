# 🔧 CodeForge Frontend

AI-Powered Code Optimization Platform - Streamlit Frontend

## 📋 Overview

CodeForge is a comprehensive code optimization platform that combines:
- **Rule-Based Analysis**: AST parsing and pattern detection
- **Semantic Understanding**: Deep code comprehension
- **AI Optimization**: Gemini 2.5 Flash integration
- **Safety Validation**: Ensuring functional correctness
- **Performance Benchmarking**: Real-time measurement

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Access to the CodeForge backend (FastAPI)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/jeremyyyS/CodeForge.git
cd CodeForge/frontend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**

Create a `.env` file in the root directory:
```env
# Optional: Custom user credentials (JSON format)
APP_USERS_JSON=[{"u":"admin","p":"admin123","role":"admin"},{"u":"user","p":"user123","role":"user"}]

# Backend URL (if different from default)
BACKEND_URL=http://localhost:8000
```

4. **Start the backend server** (in a separate terminal)
```bash
cd ../backend
python jeremy_final.py
```

5. **Run the Streamlit app**
```bash
streamlit run Home.py
```

6. **Open your browser**
Navigate to: `http://localhost:8501`

## 📁 Project Structure

```
CodeForge/
├── Home.py                 # Login page (entry point)
├── pages/
│   ├── 1_Dashboard.py      # Main optimization interface
│   ├── 2_History.py        # Optimization history
│   ├── 3_Api_Docs.py       # API documentation
│   ├── 4_About.py          # About page
│   ├── 5_Upload_Code.py    # File upload functionality
│   ├── 6_Dataset_Tools.py  # Dataset management (admin)
│   └── 7_Admin_Login.py    # Admin control panel
├── utils/
│   ├── api.py              # Backend API client
│   └── auth.py             # Authentication utilities
├── assets/
│   └── examples/
│       └── fib_unoptimized.py  # Example code
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
└── README.md              # This file
```

## 🔑 Login Credentials

### Default Users

**Admin Account:**
- Username: `admin`
- Password: `admin123`
- Access: Full system access

**Regular User:**
- Username: `user`
- Password: `user123`
- Access: Standard features

## 🎯 Features

### For All Users

1. **Code Optimization**
   - Paste code directly or upload .py files
   - Choose between AI-Powered (Hybrid) or Rules-Only modes
   - View side-by-side comparison
   - Download optimized code

2. **Performance Metrics**
   - Runtime improvements
   - Memory usage reduction
   - Speedup factors
   - Visual charts and graphs

3. **Analysis**
   - Detected optimization opportunities
   - Applied transformations
   - AI-generated explanations
   - Safety validation results

4. **History**
   - Track past optimizations
   - Filter and search
   - Export to CSV
   - View statistics

### Admin-Only Features

5. **Dataset Management**
   - Upload training datasets
   - Validate dataset quality
   - View and manage existing datasets
   - Dataset statistics

6. **System Administration**
   - User management
   - System configuration
   - Analytics dashboard
   - Maintenance tools

## 🔧 Backend Integration

### API Endpoints

The frontend communicates with these backend endpoints:

- `GET /` - Health check
- `POST /optimize` - AI-powered hybrid optimization
- `POST /optimize-rules-only` - Rule-based optimization with benchmarks
- `POST /optimize-rules-only/simple` - Fast rule-based optimization
- `POST /upload` - File upload for optimization

### Configuration

Update backend URL in `utils/api.py` if needed:
```python
BACKEND_URL = "http://localhost:8000"
```

Or set via environment variable:
```env
BACKEND_URL=http://your-backend-url:port
```

## 📊 Usage Examples

### Basic Optimization

1. Login with your credentials
2. Navigate to Dashboard
3. Paste your Python code
4. Select optimization mode
5. Click "Optimize Code"
6. Review results and download

### File Upload

1. Go to "Upload Code File"
2. Choose a .py file
3. Select optimization options
4. Click "Optimize File"
5. Download optimized version

### View History

1. Navigate to History page
2. Filter by mode or date
3. Search by Job ID
4. Export results as CSV

## 🛡️ Safety Features

- **Input Validation**: Code size limits and syntax checking
- **Safety Analysis**: Detects potential issues in optimizations
- **Confidence Scoring**: Transparency about optimization quality
- **Syntax Verification**: Ensures generated code is valid

## 📈 Performance

- **Rule-Based Mode**: < 1 second for most optimizations
- **AI-Powered Mode**: 5-30 seconds (depends on code complexity)
- **Benchmarking**: Adds 1-5 seconds for performance measurement

## 🔍 Troubleshooting

### Backend Connection Issues

```
Error: Backend not responding
```

**Solution:**
1. Check if backend is running: `python jeremy_final.py`
2. Verify backend URL in configuration
3. Check firewall settings

### Login Issues

```
Error: Invalid credentials
```

**Solution:**
1. Use default credentials (see above)
2. Check `.env` file configuration
3. Contact admin for account issues

### Performance Issues

```
Optimization timing out
```

**Solution:**
1. Reduce code size (max 10,000 characters)
2. Use Rules-Only mode for faster results
3. Check backend server resources

## 📚 API Documentation

Full API documentation is available in the app:
1. Login to the platform
2. Navigate to "API Docs"
3. View endpoints, schemas, and examples

## 🤝 Contributing

This is an academic project for:
- **Department**: AI & Data Science
- **Institution**: Rajagiri School of Engineering & Technology

## 📄 License

Academic Project - All Rights Reserved

## 🙏 Acknowledgments

- Google Gemini AI Team
- FastAPI Framework
- Streamlit Community
- Rajagiri School of Engineering & Technology
- Our Faculty Advisors

## 📧 Contact

For questions or issues:
- GitHub: [https://github.com/jeremyyyS/CodeForge](https://github.com/jeremyyyS/CodeForge)
- University Portal: Contact through official channels

## 🔄 Version History

- **v1.0.0** (Feb 2025)
  - Initial release
  - AI-powered optimization
  - Rule-based optimization
  - Complete UI/UX
  - Admin features
  - Dataset management

---

**Built with ❤️ for better code**
