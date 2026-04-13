from typing import Any, Dict, List


def filter_predictions(predictions: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Stub post-processing implementation."""
    return [p for p in predictions if p.get("confidence", 0) >= threshold]
