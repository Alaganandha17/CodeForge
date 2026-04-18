import ast
import numpy as np

# We attempt to load SentenceTransformer safely
try:
    from sentence_transformers import SentenceTransformer
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False


class SemanticPatternDetector:
    def __init__(self):
        """
        Safe initialization.
        If model fails (offline / no download / missing dependency),
        semantic detection is silently disabled.
        """

        self.enabled = False
        self.model = None

        self.inefficient_patterns = [
            "for i in range(len(array)): result.append(array[i])",
            "index = 0\nwhile index < len(array): item = array[index]; index += 1",
            "text = ''\nfor item in items: text = text + str(item)",
            "for key in dict: if key == target: return dict[key]",
            "for i in range(len(list1)): for j in range(len(list2)): compare(list1[i], list2[j])"
        ]

        if MODEL_AVAILABLE:
            try:
                self.model = SentenceTransformer("microsoft/codebert-base")
                self.pattern_embeddings = self.model.encode(
                    self.inefficient_patterns,
                    convert_to_numpy=True
                )
                self.enabled = True
            except Exception:
                self.enabled = False

    # ----------------------------------------------------
    # Extract loops safely
    # ----------------------------------------------------
    def extract_code_blocks(self, code: str):
        try:
            tree = ast.parse(code)
            blocks = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.For, ast.While)):
                    try:
                        blocks.append(ast.unparse(node))
                    except Exception:
                        continue

            return blocks
        except Exception:
            return []

    # ----------------------------------------------------
    # Main detection logic
    # ----------------------------------------------------
    def find_semantic_patterns(self, code: str, threshold=0.75):
        """
        Returns detected inefficient semantic patterns.
        If model is unavailable, returns empty list.
        """

        if not self.enabled:
            return []

        blocks = self.extract_code_blocks(code)
        if not blocks:
            return []

        try:
            block_embeddings = self.model.encode(
                blocks,
                convert_to_numpy=True
            )
        except Exception:
            return []

        detected = []

        for i, block_emb in enumerate(block_embeddings):

            # Safe cosine similarity
            pattern_norms = np.linalg.norm(self.pattern_embeddings, axis=1)
            block_norm = np.linalg.norm(block_emb)

            if block_norm == 0:
                continue

            similarities = np.dot(self.pattern_embeddings, block_emb) / (
                pattern_norms * block_norm + 1e-8
            )

            max_sim_idx = int(np.argmax(similarities))
            max_similarity = float(similarities[max_sim_idx])

            if max_similarity > threshold:
                detected.append({
                    "rule": "semantic_pattern_match",
                    "line": 0,
                    "message": f"Semantically similar to known inefficient pattern ({max_similarity:.2%} similarity)",
                    "suggestion": self._get_suggestion(max_sim_idx),
                    "confidence": round(max_similarity, 4),
                    "matched_pattern": self.inefficient_patterns[max_sim_idx]
                })

        return detected

    # ----------------------------------------------------
    # Suggestion mapping
    # ----------------------------------------------------
    def _get_suggestion(self, pattern_idx):
        suggestions = [
            "Use list comprehension or direct slicing",
            "Use enumerate() or direct iteration",
            "Use ''.join() for string concatenation",
            "Use dict.get() or membership check",
            "Use vectorized operations or reduce nested loops"
        ]

        if pattern_idx < len(suggestions):
            return suggestions[pattern_idx]

        return "Consider refactoring this pattern"


