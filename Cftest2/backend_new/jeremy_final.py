## run this file first always
import logging
import time
from collections import defaultdict
from ai_explainer import generate_ai_explanation
from semantic_search import SemanticPatternDetector
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import ast
from datetime import datetime

from rules_engine import RuleBasedOptimizer
from rule_transformer import apply_rule_based_optimizations
from llm_optimizer import optimize_with_gemini
from utils import robust_benchmark
from safety import SafetyGuard
from metrics import calculate_confidence, generate_explainability
from complexity import analyze_complexity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("codeforge")

app = FastAPI(
    title="CodeForge API",
    description="AI-powered Python code optimization platform",
    version="2.1",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiting state
_rate_limit_store: dict = defaultdict(list)
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQUESTS = 30

# File upload limits
MAX_UPLOAD_SIZE = 100 * 1024  # 100 KB

rule_optimizer = RuleBasedOptimizer()
semantic_detector = SemanticPatternDetector()
safety_guard = SafetyGuard()


class CodeRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10000)


# --------------------------------------------------------
# RATE LIMITING MIDDLEWARE
# --------------------------------------------------------
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip]
        if now - t < RATE_LIMIT_WINDOW
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning("Rate limit exceeded for %s", client_ip)
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please try again later."}
        )

    _rate_limit_store[client_ip].append(now)

    logger.info("%s %s from %s", request.method, request.url.path, client_ip)
    start_time = time.time()
    response = await call_next(request)
    duration = round(time.time() - start_time, 3)
    logger.info("%s %s completed in %ss (status %d)",
                request.method, request.url.path, duration, response.status_code)

    return response


# --------------------------------------------------------
# SAFE BENCHMARK WRAPPER
# --------------------------------------------------------
def safe_benchmark(code: str):
    try:
        return robust_benchmark(code, runs=3)
    except Exception as e:
        logger.error("Benchmark failed: %s", e)
        return None


def compute_speedup(original_bench, optimized_bench):
    if (
        original_bench
        and optimized_bench
        and original_bench.get("runtime_ms")
        and optimized_bench.get("runtime_ms")
        and optimized_bench["runtime_ms"] != 0
    ):
        return original_bench["runtime_ms"] / optimized_bench["runtime_ms"]
    return None


async def _build_response(mode, req_code, optimized, rules, transformations=None):
    """Shared async response builder — includes AI explanation every time."""
    original_bench = safe_benchmark(req_code)
    optimized_bench = safe_benchmark(optimized)

    speedup = compute_speedup(original_bench, optimized_bench)
    speedup_factor = round(speedup, 2) if speedup else 1.0

    variance_pct = original_bench.get("variance_pct", 0.0) if original_bench else 0.0
    mem_before = original_bench.get("memory_mb", 0.0) if original_bench else 0.0
    mem_after = optimized_bench.get("memory_mb", 0.0) if optimized_bench else 0.0

    safety_analysis = safety_guard.validate(
        req_code, optimized, speedup or 1.0, mem_before, mem_after
    )
    confidence = calculate_confidence(rules, speedup or 1.0, variance_pct)
    explainability = generate_explainability(
        req_code, optimized, speedup or 1.0, rules
    )

    orig_complexity = analyze_complexity(req_code)
    opt_complexity = analyze_complexity(optimized)

    # Always generate AI explanation here — never outside this function
    ai_explanation = await generate_ai_explanation(
        req_code, optimized, rules, speedup_factor
    )

    return {
        "mode": mode,
        "status": "success",
        "original_code": req_code,
        "optimized_code": optimized,
        "rules_detected": rules,
        "transformations": transformations or [],
        "benchmarks": {
            "original": original_bench,
            "optimized": optimized_bench,
            "speedup_factor": speedup_factor
        },
        "safety_analysis": safety_analysis,
        "confidence": confidence,
        "explainability": explainability,
        "complexity": {
            "original": orig_complexity,
            "optimized": opt_complexity,
        },
        "ai_explanation": ai_explanation,
        "timestamp": datetime.now().isoformat()
    }


# --------------------------------------------------------
# OFFLINE (RULES ONLY FULL)
# --------------------------------------------------------
@app.post("/optimize-rules-only")
async def optimize_rules_only(req: CodeRequest):

    rules = rule_optimizer.analyze(req.code)
    optimized, transformations = apply_rule_based_optimizations(req.code, rules)

    try:
        ast.parse(optimized)
    except SyntaxError:
        optimized = req.code

    result = await _build_response("RULES_ONLY", req.code, optimized, rules, transformations)
    return result


# --------------------------------------------------------
# OFFLINE (SIMPLE)
# --------------------------------------------------------
@app.post("/optimize-rules-only/simple")
async def optimize_rules_only_simple(req: CodeRequest):

    rules = rule_optimizer.analyze(req.code)
    optimized, _ = apply_rule_based_optimizations(req.code, rules)

    try:
        ast.parse(optimized)
    except SyntaxError:
        optimized = req.code

    return {
        "mode": "RULES_SIMPLE",
        "status": "success",
        "original_code": req.code,
        "optimized_code": optimized,
        "rules_detected": rules,
        "timestamp": datetime.now().isoformat()
    }


# --------------------------------------------------------
# ONLINE (HYBRID WITH AUTO FALLBACK)
# --------------------------------------------------------
@app.post("/optimize")
async def optimize_hybrid(req: CodeRequest):

    rules = rule_optimizer.analyze(req.code)

    try:
        semantic_patterns = semantic_detector.find_semantic_patterns(req.code)
    except Exception:
        semantic_patterns = []

    rules = rules + semantic_patterns

    try:
        optimized = await optimize_with_gemini(req.code, hints=rules)
        mode = "HYBRID"
    except Exception as e:
        logger.warning("LLM optimization failed, falling back to rules: %s", e)
        optimized, _ = apply_rule_based_optimizations(req.code, rules)
        mode = "RULES_FALLBACK"

    try:
        ast.parse(optimized)
    except SyntaxError:
        optimized = req.code

    _, transformations = apply_rule_based_optimizations(req.code, rules)
    result = await _build_response(mode, req.code, optimized, rules, transformations)
    return result


# --------------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------------
@app.post("/upload")
async def upload_code(file: UploadFile = File(...)):

    if not file.filename or not file.filename.endswith(".py"):
        raise HTTPException(400, detail="Only .py files allowed")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // 1024} KB"
        )

    try:
        code = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, detail="File must be valid UTF-8 encoded text")

    try:
        ast.parse(code)
    except SyntaxError as e:
        raise HTTPException(400, detail=f"File contains invalid Python syntax: {e}")

    return await optimize_hybrid(CodeRequest(code=code))


# --------------------------------------------------------
# HEALTH
# --------------------------------------------------------
@app.get("/")
async def root():
    from config import GEMINI_API_KEY
    return {
        "message": "CodeForge Code Optimizer",
        "status": "running",
        "version": "2.1",
        "gemini_available": bool(GEMINI_API_KEY),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("jeremy_final:app", host="0.0.0.0", port=8000, reload=True)
