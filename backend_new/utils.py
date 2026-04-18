# utils.py
import ast
import timeit
import tracemalloc
import statistics
import sys
import platform
from io import StringIO
from config import BENCHMARK_RUNS, BENCHMARK_ITERATIONS

# signal.SIGALRM is only available on Unix - use threading timeout on Windows
_IS_WINDOWS = platform.system() == "Windows"
if not _IS_WINDOWS:
    import signal

# Dangerous modules that should not be importable during benchmarking
BLOCKED_IMPORTS = {
    "os", "subprocess", "shutil", "pathlib", "socket", "http",
    "urllib", "ftplib", "smtplib", "ctypes", "multiprocessing",
    "signal", "importlib", "sys", "code", "codeop",
    "compileall", "webbrowser", "antigravity", "turtle",
    "pickle", "shelve", "marshal", "dbm", "sqlite3",
}

# Builtins that could be used for sandbox escape
BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__", "open",
    "breakpoint", "exit", "quit",
}

BENCHMARK_TIMEOUT = 10  # seconds per run


def _validate_code_safety(code: str) -> list:
    """
    Static analysis to block dangerous operations before execution.
    Returns a list of violation messages. Empty list means safe.
    """
    violations = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import: '{alias.name}'")

        if isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if root_module in BLOCKED_IMPORTS:
                    violations.append(f"Blocked import from: '{node.module}'")

        # Block dangerous function calls
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_BUILTINS:
                violations.append(f"Blocked builtin call: '{func.id}()'")
            if isinstance(func, ast.Attribute) and func.attr in (
                "system", "popen", "exec", "eval",
                "remove", "rmdir", "unlink",
            ):
                violations.append(f"Blocked method call: '.{func.attr}()'")

        # Block dangerous dunder attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                allowed_dunders = {
                    "__init__", "__str__", "__repr__", "__len__", "__iter__",
                    "__next__", "__getitem__", "__setitem__", "__contains__",
                    "__eq__", "__ne__", "__lt__", "__gt__", "__le__", "__ge__",
                    "__hash__", "__bool__", "__add__", "__sub__", "__mul__",
                    "__truediv__", "__floordiv__", "__mod__", "__pow__",
                }
                if node.attr not in allowed_dunders:
                    violations.append(f"Blocked dunder access: '{node.attr}'")

    return violations


def _make_safe_globals():
    """Create a restricted globals dict for code execution."""
    safe_names = {
        "range", "len", "int", "float", "str", "list", "dict", "set",
        "tuple", "bool", "print", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "min", "max", "sum", "abs", "round",
        "isinstance", "issubclass", "type", "hasattr", "getattr",
        "setattr", "any", "all", "iter", "next", "chr", "ord",
        "hex", "bin", "oct", "format", "repr", "hash", "id",
        "callable", "ValueError", "TypeError", "KeyError",
        "IndexError", "StopIteration", "Exception",
        "True", "False", "None",
    }

    if isinstance(__builtins__, dict):
        safe_builtins = {
            k: v for k, v in __builtins__.items()
            if k not in BLOCKED_BUILTINS
        }
    else:
        safe_builtins = {}
        for name in safe_names:
            val = getattr(__builtins__, name, None)
            if val is not None:
                safe_builtins[name] = val

    return {"__builtins__": safe_builtins}


class BenchmarkTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise BenchmarkTimeout("Benchmark execution timed out")


def _run_with_timeout(func, timeout_sec):
    """Run a function with a timeout, cross-platform."""
    if _IS_WINDOWS:
        import threading
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout_sec)

        if thread.is_alive():
            raise BenchmarkTimeout("Benchmark execution timed out")
        if exception[0]:
            raise exception[0]
        return result[0]
    else:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_sec)
        try:
            result = func()
            signal.alarm(0)
            return result
        except:
            signal.alarm(0)
            raise
        finally:
            signal.signal(signal.SIGALRM, old_handler)


def robust_benchmark(code: str, runs: int = None, iterations: int = None):
    """
    Benchmark code execution time and memory usage.
    Uses a shared namespace dict so variables persist across lines within exec(),
    fixing the Python 3 list-comprehension scope issue.
    """
    if runs is None:
        runs = BENCHMARK_RUNS
    if iterations is None:
        iterations = BENCHMARK_ITERATIONS

    # Static safety check before executing any code
    violations = _validate_code_safety(code)
    if violations:
        return {
            "runtime_ms": 0,
            "memory_mb": 0,
            "runs": 0,
            "error": f"Code blocked for safety: {'; '.join(violations)}",
            "variance_pct": 0.0
        }

    safe_globals = _make_safe_globals()
    samples = []
    mem_samples = []

    for run_num in range(runs):
        old_stdout = sys.stdout
        try:
            sys.stdout = StringIO()
            tracemalloc.start()

            t = _run_with_timeout(
                lambda: timeit.timeit(
                    lambda: exec(code, safe_globals.copy(), {}),
                    number=iterations
                ),
                BENCHMARK_TIMEOUT
            )

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            sys.stdout = old_stdout

            samples.append(t * 1000 / iterations)
            mem_samples.append(peak / (1024 ** 2))
        except BenchmarkTimeout:
            sys.stdout = old_stdout
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            return {
                "runtime_ms": 0,
                "memory_mb": 0,
                "runs": 0,
                "error": "Code execution timed out",
                "variance_pct": 0.0
            }
        except Exception:
            sys.stdout = old_stdout
            if tracemalloc.is_tracing():
                tracemalloc.stop()

    if not samples:
        return None

    mean_val = statistics.mean(samples)
    variance_pct = 0.0
    if len(samples) > 1 and mean_val > 0:
        variance_pct = round((statistics.stdev(samples) / mean_val) * 100, 2)

    return {
        "runtime_ms": round(mean_val, 3),
        "memory_mb": round(max(mem_samples), 2),
        "runs": len(samples),
        "variance_pct": variance_pct
    }
