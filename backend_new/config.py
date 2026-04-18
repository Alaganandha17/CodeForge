import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")

# Warn but don't crash - rules-only mode works without API key
if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY not found in environment. "
        "AI-powered optimization will be disabled. "
        "Rules-only mode will still work. "
        "Create a .env file with your API key for full functionality."
    )

# API Settings
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
MAX_CODE_LENGTH = int(os.getenv("MAX_CODE_LENGTH", "10000"))

# Benchmark Settings
BENCHMARK_RUNS = int(os.getenv("BENCHMARK_RUNS", "3"))
BENCHMARK_ITERATIONS = int(os.getenv("BENCHMARK_ITERATIONS", "50"))

# Safety Thresholds
MICRO_OPTIMIZATION_THRESHOLD = 1.05  # 5% speedup minimum
CODE_GROWTH_THRESHOLD = 1.2  # 20% max code growth
MEMORY_BLOAT_THRESHOLD = 1.5  # 50% max memory increase
COMPLEXITY_THRESHOLD = 1.3  # 30% max complexity increase
