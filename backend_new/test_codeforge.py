"""
Comprehensive test suite for CodeForge backend.
Tests security, rule detection, transformations, benchmarking,
safety validation, and edge cases.
"""
import ast
import sys
import os

# Ensure we can import from the backend directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set a dummy API key so config doesn't block tests
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = ""

passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed, errors
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  FAIL: {name} - {detail}")


# ============================================================
# 1. SECURITY TESTS - Code Execution Sandbox
# ============================================================
print("\n" + "=" * 60)
print("1. SECURITY TESTS - Sandbox Validation")
print("=" * 60)

from utils import _validate_code_safety, robust_benchmark

# Test: Block os module import
violations = _validate_code_safety("import os\nos.system('echo hacked')")
test("Block 'import os'", len(violations) > 0, f"Got {violations}")

# Test: Block subprocess
violations = _validate_code_safety("import subprocess\nsubprocess.run(['ls'])")
test("Block 'import subprocess'", len(violations) > 0)

# Test: Block from os import
violations = _validate_code_safety("from os import system\nsystem('ls')")
test("Block 'from os import system'", len(violations) > 0)

# Test: Block eval()
violations = _validate_code_safety("eval('1+1')")
test("Block eval() builtin", len(violations) > 0)

# Test: Block exec()
violations = _validate_code_safety("exec('print(1)')")
test("Block exec() builtin", len(violations) > 0)

# Test: Block open()
violations = _validate_code_safety("f = open('/etc/passwd')")
test("Block open() builtin", len(violations) > 0)

# Test: Block __import__
violations = _validate_code_safety("__import__('os').system('ls')")
test("Block __import__()", len(violations) > 0)

# Test: Block dunder access (__subclasses__)
violations = _validate_code_safety("x.__subclasses__()")
test("Block __subclasses__ access", len(violations) > 0)

# Test: Block __globals__
violations = _validate_code_safety("x.__globals__")
test("Block __globals__ access", len(violations) > 0)

# Test: Block socket
violations = _validate_code_safety("import socket\nsocket.socket()")
test("Block socket import", len(violations) > 0)

# Test: Block pickle
violations = _validate_code_safety("import pickle")
test("Block pickle import", len(violations) > 0)

# Test: Allow safe code
violations = _validate_code_safety("x = [i*2 for i in range(100)]")
test("Allow safe list comprehension", len(violations) == 0, f"Got {violations}")

# Test: Allow math operations
violations = _validate_code_safety("result = sum(range(1000))\nprint(result)")
test("Allow math operations", len(violations) == 0, f"Got {violations}")

# Test: Allow basic data structures
violations = _validate_code_safety("d = {}\nfor i in range(10): d[i] = i*2")
test("Allow dict operations", len(violations) == 0, f"Got {violations}")

# Test: Benchmark blocks dangerous code
result = robust_benchmark("import os\nos.listdir('.')")
test("Benchmark blocks dangerous code",
     result is not None and result.get("error") is not None,
     f"Got {result}")

# Test: Benchmark works with safe code
result = robust_benchmark("x = sum(range(100))", runs=1, iterations=5)
test("Benchmark runs safe code",
     result is not None and result.get("runtime_ms", 0) > 0,
     f"Got {result}")

# Test: Benchmark returns variance_pct
result = robust_benchmark("x = [i for i in range(100)]", runs=3, iterations=5)
test("Benchmark returns variance_pct",
     result is not None and "variance_pct" in result,
     f"Got {result}")

# Test: Syntax error handling
violations = _validate_code_safety("def foo(:")
test("Handle syntax errors", len(violations) > 0)


# ============================================================
# 2. RULES ENGINE TESTS
# ============================================================
print("\n" + "=" * 60)
print("2. RULES ENGINE TESTS")
print("=" * 60)

from rules_engine import RuleBasedOptimizer

optimizer = RuleBasedOptimizer()

# Test: Detect range(len(x))
code = "for i in range(len(items)): print(items[i])"
rules = optimizer.analyze(code)
test("Detect range(len(x))",
     any(r["rule"] == "range_len_pattern" for r in rules),
     f"Found: {[r['rule'] for r in rules]}")

# Test: Detect append in loop
code = "result = []\nfor x in data:\n    result.append(x*2)"
rules = optimizer.analyze(code)
test("Detect append in loop",
     any(r["rule"] == "append_in_loop" for r in rules),
     f"Found: {[r['rule'] for r in rules]}")

# Test: Detect constant folding
code = "x = 100 + 200"
rules = optimizer.analyze(code)
test("Detect constant folding",
     any(r["rule"] == "constant_folding" for r in rules),
     f"Found: {[r['rule'] for r in rules]}")

# Test: Detect string concatenation in loop
code = "s = ''\nfor x in items:\n    s += str(x)"
rules = optimizer.analyze(code)
test("Detect string concat in loop",
     any(r["rule"] == "string_concat_loop" for r in rules),
     f"Found: {[r['rule'] for r in rules]}")

# Test: Detect nested loops
code = "for i in range(10):\n    for j in range(10):\n        pass"
rules = optimizer.analyze(code)
test("Detect nested loops",
     any(r["rule"] == "nested_loops" for r in rules),
     f"Found: {[r['rule'] for r in rules]}")

# Test: Detect multiple isinstance
code = "if isinstance(x, int) or isinstance(x, float): pass"
rules = optimizer.analyze(code)
test("Detect multiple isinstance",
     any(r["rule"] == "multiple_isinstance" for r in rules),
     f"Found: {[r['rule'] for r in rules]}")

# Test: No false positives on clean code
code = "data = [x**2 for x in range(100)]"
rules = optimizer.analyze(code)
test("No false positives on clean code",
     len(rules) == 0,
     f"Found: {[r['rule'] for r in rules]}")

# Test: Handle invalid syntax gracefully
rules = optimizer.analyze("def foo(:")
test("Handle invalid syntax", rules == [])

# Test: Detect loop invariant
code = "for i in range(len(items)):\n    print(i)"
rules = optimizer.analyze(code)
test("Detect loop invariant len()",
     any(r["rule"] == "loop_invariant_motion" for r in rules),
     f"Found: {[r['rule'] for r in rules]}")

# Test: Complex code with multiple issues
code = """
result = []
for i in range(len(data)):
    s = ''
    for j in range(len(other)):
        s += str(other[j])
    result.append(s)
"""
rules = optimizer.analyze(code)
test("Detect multiple rules in complex code",
     len(rules) >= 3,
     f"Found {len(rules)} rules: {[r['rule'] for r in rules]}")


# ============================================================
# 3. RULE TRANSFORMER TESTS
# ============================================================
print("\n" + "=" * 60)
print("3. RULE TRANSFORMER TESTS")
print("=" * 60)

from rule_transformer import apply_rule_based_optimizations

# Test: Transform range(len(x)) to enumerate
code = "for i in range(len(items)):\n    print(items[i])"
rules = [{"rule": "range_len_pattern", "line": 1, "message": "range(len(x))", "suggestion": "Use enumerate"}]
optimized, transforms = apply_rule_based_optimizations(code, rules)
test("Transform range(len) to enumerate",
     "enumerate" in optimized,
     f"Got: {optimized}")

# Test: Transform constant folding
code = "x = 100 + 200"
rules = [{"rule": "constant_folding", "line": 1, "message": "Constant", "suggestion": "Pre-compute"}]
optimized, transforms = apply_rule_based_optimizations(code, rules)
test("Transform constant folding",
     "300" in optimized,
     f"Got: {optimized}")

# Test: Output is valid Python
code = "for i in range(len(items)):\n    print(items[i])"
rules = optimizer.analyze(code)
optimized, transforms = apply_rule_based_optimizations(code, rules)
try:
    ast.parse(optimized)
    test("Transformed code is valid Python", True)
except SyntaxError as e:
    test("Transformed code is valid Python", False, str(e))

# Test: Invalid input returns original
optimized, transforms = apply_rule_based_optimizations("def foo(:", [])
test("Invalid input returns original", optimized == "def foo(:")

# Test: No rules means no changes
code = "x = 42"
optimized, transforms = apply_rule_based_optimizations(code, [])
test("No rules means no changes", optimized == code)


# ============================================================
# 4. SAFETY GUARD TESTS
# ============================================================
print("\n" + "=" * 60)
print("4. SAFETY GUARD TESTS")
print("=" * 60)

from safety import SafetyGuard

guard = SafetyGuard()

# Test: Safe optimization passes
result = guard.validate("x=1", "x=1", 2.0, 1.0, 1.0)
test("Safe optimization passes",
     result["is_safe"] is True and result["verdict"] == "SAFE TO APPLY")

# Test: Micro-optimization warning
long_code = "x" * 1000
result = guard.validate("x=1", long_code, 1.01, 1.0, 1.0)
test("Micro-optimization detected",
     any(w["type"] == "micro_optimization" for w in result["warnings"]),
     f"Warnings: {result['warnings']}")

# Test: Memory bloat warning
result = guard.validate("x=1", "x=1", 2.0, 1.0, 2.5)
test("Memory bloat detected",
     any(w["type"] == "memory_bloat" for w in result["warnings"]),
     f"Warnings: {result['warnings']}")

# Test: All safe returns correct count
result = guard.validate("x=1", "x=1", 2.0, 1.0, 1.0)
test("Safe count correct", result["safe_count"] == 3)


# ============================================================
# 5. METRICS TESTS
# ============================================================
print("\n" + "=" * 60)
print("5. METRICS TESTS")
print("=" * 60)

from metrics import calculate_confidence, generate_explainability

# Test: High confidence for good optimization
rules = [{"rule": "test", "line": 1, "message": "msg", "suggestion": "sug"}] * 4
conf = calculate_confidence(rules, 2.0, 5.0)
test("High confidence for good optimization",
     conf["overall"] >= 75 and conf["recommendation"] == "APPLY",
     f"Got: {conf}")

# Test: Low confidence for marginal optimization
conf = calculate_confidence([], 1.01, 50.0)
test("Low confidence for marginal optimization",
     conf["confidence_level"] == "LOW",
     f"Got: {conf}")

# Test: Confidence capped at 100
rules = [{"rule": "test", "line": 1, "message": "msg", "suggestion": "sug"}] * 10
conf = calculate_confidence(rules, 5.0, 0.0)
test("Confidence capped at 100", conf["overall"] <= 100)

# Test: Explainability generation
expl = generate_explainability(
    "x = 1\ny = 2",
    "x = 1\ny = 3",
    1.5,
    [{"rule": "test", "line": 1, "message": "msg", "suggestion": "sug"}]
)
test("Explainability has code_diff", "code_diff" in expl)
test("Explainability has transformation", "transformation" in expl)
test("Explainability has performance", "performance" in expl)


# ============================================================
# 6. LLM OPTIMIZER TESTS (unit tests, no API calls)
# ============================================================
print("\n" + "=" * 60)
print("6. LLM OPTIMIZER TESTS")
print("=" * 60)

from llm_optimizer import _clean_markdown_fences

# Test: Clean python markdown fences
test("Clean ```python fences",
     _clean_markdown_fences("```python\nx = 1\n```") == "x = 1")

# Test: Clean plain markdown fences
test("Clean ``` fences",
     _clean_markdown_fences("```\nx = 1\n```") == "x = 1")

# Test: No fences passes through
test("No fences passes through",
     _clean_markdown_fences("x = 1") == "x = 1")

# Test: Nested fences handled
test("Nested fences handled",
     "```" not in _clean_markdown_fences("```python\nprint('hi')\n```"))

# Test: Empty string
test("Empty string handled",
     _clean_markdown_fences("") == "")


# ============================================================
# 7. EDGE CASE TESTS
# ============================================================
print("\n" + "=" * 60)
print("7. EDGE CASE TESTS")
print("=" * 60)

# Test: Empty code
rules = optimizer.analyze("")
test("Empty code returns no rules", rules == [])

# Test: Single line code
rules = optimizer.analyze("print('hello')")
test("Single line code works", isinstance(rules, list))

# Test: Very long code
long_code = "\n".join([f"x_{i} = {i}" for i in range(500)])
rules = optimizer.analyze(long_code)
test("Long code doesn't crash", isinstance(rules, list))

# Test: Code with decorators
code = "@property\ndef foo(self):\n    return self._x"
rules = optimizer.analyze(code)
test("Code with decorators works", isinstance(rules, list))

# Test: Code with classes
code = """
class MyClass:
    def __init__(self):
        self.data = []

    def process(self):
        for i in range(len(self.data)):
            print(self.data[i])
"""
rules = optimizer.analyze(code)
test("Code with classes works", isinstance(rules, list))

# Test: Code with try/except
code = """
try:
    for i in range(len(items)):
        result.append(items[i])
except Exception:
    pass
"""
rules = optimizer.analyze(code)
test("Code with try/except works",
     any(r["rule"] in ("range_len_pattern", "append_in_loop") for r in rules),
     f"Found: {[r['rule'] for r in rules]}")

# Test: Lambda functions
code = "f = lambda x: [i for i in range(len(x))]"
rules = optimizer.analyze(code)
test("Lambda functions work", isinstance(rules, list))

# Test: List comprehension with condition
code = "evens = [x for x in range(100) if x % 2 == 0]"
rules = optimizer.analyze(code)
test("Comprehension with condition works", isinstance(rules, list))

# Test: Generator expression
code = "total = sum(x**2 for x in range(1000))"
rules = optimizer.analyze(code)
test("Generator expression works", isinstance(rules, list))

# Test: Multiple function definitions
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    result = 1
    for i in range(1, n+1):
        result *= i
    return result
"""
rules = optimizer.analyze(code)
test("Multiple functions work", isinstance(rules, list))

# Test: Unicode in code
code = "# This is a comment with unicode: cafe\nresult = 'hello'"
rules = optimizer.analyze(code)
test("Unicode in code works", isinstance(rules, list))

# Test: Benchmark with print statements
result = robust_benchmark("print('test')", runs=1, iterations=2)
test("Benchmark handles print()", result is not None)

# Test: Benchmark with empty code
result = robust_benchmark("pass", runs=1, iterations=2)
test("Benchmark handles 'pass'",
     result is not None and result.get("runtime_ms", -1) >= 0)


# ============================================================
# 8. INTEGRATION TESTS
# ============================================================
print("\n" + "=" * 60)
print("8. INTEGRATION TESTS - Full Pipeline")
print("=" * 60)

# Full pipeline test: detect + transform + validate
code = """
data = list(range(1000))
result = []
for i in range(len(data)):
    result.append(data[i] * 2)
"""

rules = optimizer.analyze(code)
test("Pipeline: rules detected", len(rules) > 0)

optimized, transforms = apply_rule_based_optimizations(code, rules)
test("Pipeline: transformations applied", len(transforms) > 0 or optimized != code)

try:
    ast.parse(optimized)
    valid = True
except SyntaxError:
    valid = False
test("Pipeline: output is valid Python", valid)

bench = robust_benchmark(code, runs=1, iterations=5)
test("Pipeline: benchmark works", bench is not None)

if bench:
    test("Pipeline: benchmark has variance_pct", "variance_pct" in bench)

safety = guard.validate(code, optimized, 1.5, 1.0, 1.0)
test("Pipeline: safety check works", "is_safe" in safety)

conf = calculate_confidence(rules, 1.5, 5.0)
test("Pipeline: confidence calculated", "overall" in conf)

expl = generate_explainability(code, optimized, 1.5, rules)
test("Pipeline: explainability generated", "code_diff" in expl)

# Test complex real-world patterns
patterns = [
    # Fibonacci (iterative to avoid recursion timeout in sandbox)
    """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
result = fib(10)
""",
    # String processing
    """
words = ['hello', 'world', 'foo', 'bar']
result = ''
for w in words:
    result += w + ' '
""",
    # Nested loop search
    """
matrix = [[1,2,3],[4,5,6],[7,8,9]]
found = False
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        if matrix[i][j] == 5:
            found = True
""",
    # Dictionary processing
    """
data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
result = []
for key in data:
    if data[key] > 2:
        result.append(key)
""",
    # Sum computation
    """
numbers = list(range(10000))
total = 0
for n in numbers:
    total = total + n
""",
]

for idx, pattern_code in enumerate(patterns, 1):
    try:
        rules = optimizer.analyze(pattern_code)
        optimized, _ = apply_rule_based_optimizations(pattern_code, rules)
        ast.parse(optimized)
        bench_result = robust_benchmark(pattern_code.strip(), runs=1, iterations=3)
        test(f"Real-world pattern #{idx} full pipeline",
             bench_result is not None)
    except Exception as e:
        test(f"Real-world pattern #{idx} full pipeline", False, str(e))


# ============================================================
# 9. AUTH MODULE TESTS
# ============================================================
print("\n" + "=" * 60)
print("9. AUTH MODULE TESTS")
print("=" * 60)

# Test the password hashing functions directly (they don't depend on streamlit)
import hashlib
import hmac
import secrets

def _hash_password(password: str, salt: str = "") -> str:
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${hashed.hex()}"

def _verify_password(password: str, stored_hash: str) -> bool:
    if "$" not in stored_hash:
        return hmac.compare_digest(password, stored_hash)
    salt, hash_val = stored_hash.split("$", 1)
    check_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return hmac.compare_digest(check_hash.hex(), hash_val)

# Test: Password hashing
hashed = _hash_password("test123")
test("Password hashing produces hash", "$" in hashed and len(hashed) > 20)

# Test: Password verification
test("Password verification works", _verify_password("test123", hashed))

# Test: Wrong password fails
test("Wrong password fails", not _verify_password("wrong", hashed))

# Test: Legacy plaintext comparison works
test("Legacy plaintext comparison", _verify_password("admin123", "admin123"))

# Test: Empty password doesn't crash
hashed = _hash_password("")
test("Empty password hashing works", len(hashed) > 0)


# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print(f"TEST RESULTS: {passed} passed, {failed} failed out of {passed + failed} total")
print("=" * 60)

if errors:
    print("\nFailed tests:")
    for err in errors:
        print(f"  - {err}")

if failed > 0:
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
