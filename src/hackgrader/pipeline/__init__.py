"""Gallery-grading orchestration and its stable application entry point.

The implementation lives in :mod:`hackgrader.pipeline.workflow`. Import
``run_gallery`` from this package or from the workflow module directly.
"""

from .workflow import individual_query, run_gallery

__all__ = ["individual_query", "run_gallery"]
