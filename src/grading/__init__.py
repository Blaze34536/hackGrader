"""Ollama evaluation and judge-report annotation.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from .reporting import Reporting
from .runtime import Model

__all__ = ["Model", "Reporting"]
