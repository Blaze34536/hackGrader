"""Run optimizer rounds and the ordinary gated release sequence."""

import gc

from ..training import artifacts
from .resume import require_selection


def run_training(
    *,
    hooks,
    options,
    load_runtime,
    warn_gpu,
    make_model,
    prepare,
    report,
    run_rounds,
    final,
    save,
    deploy,
    block,
    remaining,
    estimate,
    train_rows,
    validation_rows,
    test_rows,
    fresh_repository_test=False,
    continuation_checkpoint,
    resume_checkpoint,
    run_dir,
    started_at,
    run_hours,
    remaining_seconds,
    training_deadline,
    validation_deadline,
    deadline,
    minimum_vram,
    repeat,
    strict_gates: bool = False,
):
    """Train isolated optimizer rounds then release only a fully passing selection."""
    dependencies = load_runtime()
    torch = dependencies["torch"]
    warn_gpu(torch, minimum_vram)
    model, tokenizer = (
        make_model(dependencies)
        if continuation_checkpoint is None
        else hooks["_round_model"](dependencies, continuation_checkpoint)
    )
    train_dataset, eval_dataset = prepare(tokenizer, train_rows, validation_rows)
    report()
    run_rounds = (
        run_rounds or hooks.get("run_training_rounds") or hooks["run_training_rounds"]
    )
    selected = run_rounds(
        dependencies,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        train_rows * repeat,
        validation_rows,
        started_at=started_at,
        training_deadline=training_deadline,
        validation_deadline=validation_deadline,
        resume_selection=None,
        starting_checkpoint=continuation_checkpoint,
        budget_hours=(
            run_hours
            if resume_checkpoint is None
            else min(run_hours, remaining_seconds / 3600)
        ),
        run_dir=run_dir,
        strict_gates=strict_gates,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    provisional = bool(
        selected
        and selected.cap_reached
        and not selected.metrics["passed"]
        and not strict_gates
    )
    selected = require_selection(
        selected, run_dir, started_at, block, allow_failed=provisional
    )
    provisional = bool(selected.cap_reached and not selected.metrics["passed"])
    if strict_gates and provisional:
        raise SystemExit(
            "[!] Strict release mode forbids provisional outcomes; no test, merge, or registration."
        )
    if provisional:
        selected.metrics["cap_reached"] = True
        print(
            "[!] Training cap reached with validation gates still failing. "
            "A provisional tag is possible only if the one-time independent test passes."
        )
    if not selected.metrics["passed"] and not provisional:
        block(selected.metrics, checkpoint_path=selected.checkpoint_path)
        raise SystemExit(
            "[!] Validation did not pass; final test scoring and deployment stopped."
        )
    try:
        if provisional:
            model, tokenizer = final(
                dependencies,
                selected,
                test_rows,
                deadline,
                allow_provisional_validation=True,
                require_real_repository_test=fresh_repository_test,
                train_rows=train_rows,
                validation_rows=validation_rows,
            )
            if (
                not selected.provisional
                or selected.test_metrics is None
                or not selected.test_metrics.get("passed")
            ):
                raise SystemExit(
                    "[!] Provisional deployment requires a passing independent-test result."
                )
        else:
            model, tokenizer = final(
                dependencies,
                selected,
                test_rows,
                deadline,
                require_real_repository_test=fresh_repository_test,
                train_rows=train_rows,
                validation_rows=validation_rows,
            )
        remaining(deadline, "final merge and deployment")
        adapter_file = save(model, tokenizer)
        if provisional:
            deploy(
                model,
                tokenizer,
                len(train_dataset),
                adapter_file,
                deadline=deadline,
                provisional=True,
                validation_metrics=selected.metrics,
                test_metrics=selected.test_metrics,
            )
        else:
            deploy(
                model,
                tokenizer,
                len(train_dataset),
                adapter_file,
                deadline=deadline,
                validation_metrics=selected.metrics,
                test_metrics=selected.test_metrics,
            )
    except artifacts.RunDeadlineExceeded as exc:
        raise SystemExit(
            f"[!] {run_hours:.1f}-hour cap reached during {exc}; "
            "no model was merged or registered."
        ) from None
    finally:
        gc.collect()
        torch.cuda.empty_cache()
