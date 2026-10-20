"""Public entry points for the modular learning pipeline."""


def main(**options):
    """Run the capped training workflow or validation-only adapter release."""
    from .facade import main as run

    return run(**options)


def run_training_rounds(*args, **options):
    """Train and validation-score bounded optimizer rounds."""
    from ..training.loop import run_training_rounds as run

    return run(*args, **options)


__all__ = ["main", "run_training_rounds"]
