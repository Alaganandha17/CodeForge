# rule_transformer.py
import ast
import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def apply_rule_based_optimizations(code: str, rules: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Applies deterministic, offline optimizations based on detected rules.
    """
    optimized = code
    transformations = []

    try:
        ast.parse(code)
    except SyntaxError:
        return code, []

    for rule in rules:
        rule_name = rule.get("rule", "")

        if rule_name == "range_len_pattern":
            match = re.search(r"for (\w+) in range\(len\((\w+)\)\):", optimized)
            if match:
                loop_var, collection = match.groups()
                optimized = re.sub(
                    r"for " + loop_var + r" in range\(len\(" + collection + r"\)\):",
                    f"for {loop_var}, item in enumerate({collection}):",
                    optimized
                )
                optimized = re.sub(rf"{collection}\[{loop_var}\]", "item", optimized)
                transformations.append(rule)

        elif rule_name == "constant_folding":
            new_code = re.sub(
                r"\b(\d+)\s*\+\s*(\d+)\b",
                lambda m: str(int(m.group(1)) + int(m.group(2))),
                optimized
            )
            if new_code != optimized:
                optimized = new_code
                transformations.append(rule)

        elif rule_name == "append_in_loop":
            # Transform: var = []; for x in iterable: var.append(expr)
            # Into: var = [expr for x in iterable]
            pattern = re.compile(
                r"(\w+)\s*=\s*\[\]\s*\n"
                r"(\s*)for\s+(\w+)\s+in\s+(.+?):\s*\n"
                r"\s+\1\.append\((.+?)\)\s*$",
                re.MULTILINE
            )
            match = pattern.search(optimized)
            if match:
                var, indent, loop_var, iterable, expr = match.groups()
                replacement = f"{var} = [{expr} for {loop_var} in {iterable}]"
                optimized = pattern.sub(replacement, optimized, count=1)
                transformations.append(rule)

        elif rule_name == "string_concat_loop":
            # Transform: var = ""; for x in it: var += expr -> var = ''.join(...)
            pattern = re.compile(
                r"(\w+)\s*=\s*[\"\']{2}\s*\n"
                r"(\s*)for\s+(\w+)\s+in\s+(.+?):\s*\n"
                r"\s+\1\s*\+=\s*(.+?)\s*$",
                re.MULTILINE
            )
            match = pattern.search(optimized)
            if match:
                var, indent, loop_var, iterable, expr = match.groups()
                replacement = f"{var} = ''.join({expr} for {loop_var} in {iterable})"
                optimized = pattern.sub(replacement, optimized, count=1)
                transformations.append(rule)

        elif rule_name == "multiple_isinstance":
            # isinstance(x, A) or isinstance(x, B) -> isinstance(x, (A, B))
            pattern = re.compile(
                r"isinstance\((\w+),\s*(\w+)\)\s+or\s+isinstance\(\1,\s*(\w+)\)"
            )
            match = pattern.search(optimized)
            if match:
                obj, type1, type2 = match.groups()
                optimized = pattern.sub(
                    f"isinstance({obj}, ({type1}, {type2}))",
                    optimized, count=1
                )
                transformations.append(rule)

        elif rule_name == "loop_invariant_motion":
            if "enumerate" not in optimized:
                pattern = re.compile(
                    r"^(\s*)for\s+(\w+)\s+in\s+range\(len\((\w+)\)\):",
                    re.MULTILINE
                )
                match = pattern.search(optimized)
                if match:
                    indent, loop_var, collection = match.groups()
                    length_var = f"_{collection}_len"
                    optimized = pattern.sub(
                        f"{indent}{length_var} = len({collection})\n{indent}for {loop_var} in range({length_var}):",
                        optimized, count=1
                    )
                    transformations.append(rule)

    # Verify the optimized code is still valid Python
    try:
        ast.parse(optimized)
    except SyntaxError:
        logger.warning("Transformation produced invalid Python, reverting")
        return code, []

    return optimized, transformations
