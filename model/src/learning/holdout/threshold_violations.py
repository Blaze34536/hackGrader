"""Build numerical release-gate failure diagnostics."""

from ..config import (
    HOLDOUT_MAE_LIMIT,
    HOLDOUT_MIN_PRECISION,
    HOLDOUT_MIN_RECALL,
    HOLDOUT_MIN_SPECIFICITY,
    MIN_HOLDOUT_SEPARATION,
    MIN_HOLDOUT_SPREAD,
)


def threshold_violations(metrics: dict) -> list[dict[str, str]]:
    """Report every numeric release threshold the scored sample misses."""
    split = metrics["split"]
    failures = []
    checks = (
        (
            "mean_absolute_error",
            metrics["mean_absolute_error"] > HOLDOUT_MAE_LIMIT,
            f"[!] {split} mean absolute error {metrics['mean_absolute_error']:.1f} exceeds {HOLDOUT_MAE_LIMIT:.1f}; not deploying.",
        ),
        (
            "spread",
            metrics["score_spread"] < MIN_HOLDOUT_SPREAD,
            f"[!] {split} score spread {metrics['score_spread']:.1f} is below the {MIN_HOLDOUT_SPREAD:.1f} minimum; not deploying.",
        ),
        (
            "separation",
            metrics["class_separation"] < MIN_HOLDOUT_SEPARATION,
            f"[!] {split} class separation {metrics['class_separation']:.1f} is below the {MIN_HOLDOUT_SEPARATION:.1f} minimum; not deploying.",
        ),
    )
    failures.extend(
        {"gate": gate, "message": message} for gate, failed, message in checks if failed
    )
    reliability = [
        f"{name} {metrics[name]:.2f} < {minimum:.2f}"
        for name, minimum in (
            ("precision", HOLDOUT_MIN_PRECISION),
            ("recall", HOLDOUT_MIN_RECALL),
            ("specificity", HOLDOUT_MIN_SPECIFICITY),
        )
        if metrics[name] < minimum
    ]
    if reliability:
        failures.append(
            {
                "gate": "reliability",
                "message": f"[!] {split} reliability floors missed: {', '.join(reliability)}; not deploying.",
            }
        )
    return failures
