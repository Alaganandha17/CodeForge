# complexity.py
import ast
from typing import Dict


def analyze_complexity(code: str) -> Dict:
    """Analyze code complexity: cyclomatic complexity, nesting depth, Big-O estimate."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"error": "Could not parse code"}

    visitor = ComplexityVisitor()
    visitor.visit(tree)

    big_o = _estimate_big_o(tree)

    return {
        "cyclomatic_complexity": visitor.complexity,
        "max_nesting_depth": visitor.max_depth,
        "num_functions": visitor.num_functions,
        "num_loops": visitor.num_loops,
        "num_branches": visitor.num_branches,
        "lines_of_code": len([l for l in code.splitlines() if l.strip() and not l.strip().startswith('#')]),
        "big_o_estimate": big_o,
    }


class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 1  # Start at 1 for the base path
        self.max_depth = 0
        self._current_depth = 0
        self.num_functions = 0
        self.num_loops = 0
        self.num_branches = 0

    def _enter_block(self):
        self._current_depth += 1
        self.max_depth = max(self.max_depth, self._current_depth)

    def _exit_block(self):
        self._current_depth -= 1

    def visit_If(self, node):
        self.complexity += 1
        self.num_branches += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_For(self, node):
        self.complexity += 1
        self.num_loops += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_While(self, node):
        self.complexity += 1
        self.num_loops += 1
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # Each 'and'/'or' adds a decision point
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.num_functions += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.num_functions += 1
        self.generic_visit(node)


def _estimate_big_o(tree: ast.AST) -> str:
    """Estimate Big-O time complexity based on loop nesting."""
    max_loop_depth = 0

    def _walk_depth(node, depth=0):
        nonlocal max_loop_depth
        if isinstance(node, (ast.For, ast.While)):
            depth += 1
            max_loop_depth = max(max_loop_depth, depth)
        for child in ast.iter_child_nodes(node):
            _walk_depth(child, depth)

    _walk_depth(tree)

    labels = {
        0: "O(1) - Constant",
        1: "O(n) - Linear",
        2: "O(n²) - Quadratic",
        3: "O(n³) - Cubic",
    }
    if max_loop_depth in labels:
        return labels[max_loop_depth]
    return f"O(n^{max_loop_depth}) - Polynomial"
