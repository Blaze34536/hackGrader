"""Release a previously trained adapter without another optimizer run."""

from typing import Any

from .. import holdout, scoring
from ..resume import validate_resume_checkpoint
from . import final_test, models, resume


def release_forked_checkpoint(
    load_runtime,
    warn_gpu,
    report,
    checkpoint_path: str,
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    training_row_count: int,
    validation_deadline: float,
    deadline: float,
    *,
    minimum_vram: float,
    save_adapter,
    merge_and_deploy,
    blocked,
    remaining,
    require_real_repository_test: bool = False,
    train_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Gate saved round candidates, then test and register without another training run."""
    checkpoint_path = validate_resume_checkpoint(checkpoint_path)
    dependencies = load_runtime()
    warn_gpu(dependencies["torch"], minimum_vram)
    model, tokenizer = models.load_selected_model(dependencies, checkpoint_path)
    report()
    selection = resume.score_resume_round(
        dependencies,
        model,
        tokenizer,
        validation_rows,
        checkpoint_path,
        validation_deadline,
    )
    if not selection.metrics["passed"]:
        blocked(selection.metrics, checkpoint_path=selection.checkpoint_path)
        raise SystemExit(
            "[!] Validation did not pass; independent testing and deployment stopped."
        )
    del model, tokenizer
    dependencies["torch"].cuda.empty_cache()
    model, tokenizer = final_test.run_final_test(
        dependencies,
        selection,
        test_rows,
        deadline,
        load_model=models.load_selected_model,
        score=scoring.evaluate_split,
        report_gates=holdout.report_release_gates,
        summarize=holdout.summarize_holdout,
        block_release=blocked,
        require_real_repository_test=require_real_repository_test,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )
    remaining(deadline, "final merge and deployment")
    adapter_file = save_adapter(model, tokenizer)
    merge_and_deploy(
        model,
        tokenizer,
        training_row_count,
        adapter_file,
        deadline=deadline,
        validation_metrics=selection.metrics,
        test_metrics=selection.test_metrics,
    )
