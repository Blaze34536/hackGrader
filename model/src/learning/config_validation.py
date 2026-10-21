"""Validate training, data, and release-gate configuration values."""


def validate_config(settings: dict[str, object]) -> None:
    """Reject incompatible limits before data loading or model initialization."""
    if (
        min(
            settings["REPEAT"],
            settings["EPOCHS"],
            settings["MICRO_BATCH"],
            settings["ACCUM"],
        )
        < 1
    ):
        raise ValueError(
            "training repeat, epochs, batch size, and accumulation must be positive"
        )
    hours = settings["HOURS"]
    maximum = settings["MAX_TRAINING_HOURS"]
    if hours <= 0 or hours > maximum:
        raise ValueError(
            f"HACKGRADER_HOURS must be positive and no more than {maximum:g}"
        )
    validation_reserve = settings["VALIDATION_SCORING_RESERVE_HOURS"]
    test_reserve = settings["FINAL_TEST_RESERVE_HOURS"]
    if (
        min(validation_reserve, test_reserve) < 0
        or validation_reserve + test_reserve >= hours
    ):
        raise ValueError(
            "validation and final-test reserves must fit inside HACKGRADER_HOURS"
        )
    if settings["MAX_GATE_CANDIDATES"] < 1:
        raise ValueError("HACKGRADER_MAX_GATE_CANDIDATES must be positive")
    steps = settings["MAX_STEPS_PER_ROUND"]
    eval_steps = settings["EVAL_STEPS"]
    if steps < eval_steps or steps % eval_steps:
        raise ValueError(
            "HACKGRADER_MAX_STEPS_PER_ROUND must be a positive multiple of EVAL_STEPS"
        )
    positive_names = (
        "CHECK_ROWS",
        "MIN_TEST_ROWS",
        "MIN_TEST_CLASS_ROWS",
        "EVAL_ROWS",
        "EVAL_STEPS",
        "MIN_TRAIN_ROWS",
        "MIN_TRAIN_CLASS_ROWS",
        "MIN_SCORE_BAND_ROWS",
        "MIN_VALIDATION_ROWS",
        "MIN_VALIDATION_CLASS_ROWS",
        "MIN_VALIDATION_SCORE_BAND_ROWS",
        "MIN_GENERATED_TEST_ROWS",
        "MIN_GENERATED_TEST_SCORE_BAND_ROWS",
        "MIN_GENERATED_TEST_CLASS_ROWS",
    )
    if min(settings[name] for name in positive_names) < 1:
        raise ValueError(
            "dataset size, evaluation, and per-class minimums must be positive"
        )
    optional_names = (
        "MIN_BOUNDARY_CLASS_ROWS",
        "MIN_TRAIN_MIDDLE_SCORE_BAND_ROWS",
        "MIN_VALIDATION_MIDDLE_SCORE_BAND_ROWS",
        "MIN_GENERATED_TEST_MIDDLE_SCORE_BAND_ROWS",
        "MIN_VALIDATION_BOUNDARY_CLASS_ROWS",
        "MIN_GENERATED_TEST_BOUNDARY_CLASS_ROWS",
    )
    if min(settings[name] for name in optional_names) < 0:
        raise ValueError("middle and boundary score-band minimums cannot be negative")
    class_minimums = (
        ("MIN_TEST_CLASS_ROWS", "MIN_TEST_ROWS", "test"),
        ("MIN_TRAIN_CLASS_ROWS", "MIN_TRAIN_ROWS", "train"),
        ("MIN_VALIDATION_CLASS_ROWS", "MIN_VALIDATION_ROWS", "validation"),
        ("MIN_GENERATED_TEST_CLASS_ROWS", "MIN_GENERATED_TEST_ROWS", "generated test"),
    )
    for class_key, rows_key, label in class_minimums:
        if 2 * settings[class_key] > settings[rows_key]:
            raise ValueError(
                f"{label} row minimum must cover both configured class minimums"
            )
    if settings["MIN_TEST_ROWS"] > settings["MIN_GENERATED_TEST_ROWS"]:
        raise ValueError(
            "generated test row minimum must cover CHECK_ROWS and MIN_TEST_ROWS"
        )
    if settings["MIN_TEST_CLASS_ROWS"] > settings["MIN_GENERATED_TEST_CLASS_ROWS"]:
        raise ValueError(
            "generated test class minimum cannot be below MIN_TEST_CLASS_ROWS"
        )
    validation_coverage = max(
        settings[name] for name in ("CHECK_ROWS", "EVAL_ROWS", "MIN_TEST_ROWS")
    )
    if validation_coverage > settings["MIN_VALIDATION_ROWS"]:
        raise ValueError(
            "validation row minimum must cover CHECK_ROWS, EVAL_ROWS, and MIN_TEST_ROWS"
        )
    if settings["CHECK_ROWS"] > settings["MIN_GENERATED_TEST_ROWS"]:
        raise ValueError(
            "independent-test minimum must cover the stratified holdout check size"
        )
    if settings["MIN_TEST_CLASS_ROWS"] > settings["MIN_VALIDATION_CLASS_ROWS"]:
        raise ValueError("validation class minimum cannot be below MIN_TEST_CLASS_ROWS")
    if settings["LORA_LR"] <= 0:
        raise ValueError("the LoRA learning rate must be positive")
    if not 0 <= settings["LORA_DROPOUT"] < 1:
        raise ValueError("the LoRA dropout must be at least 0 and below 1")
    if settings["WEIGHT_DECAY"] < 0:
        raise ValueError("the weight decay cannot be negative")
    if (
        settings["MIN_ADAPTIVE_LR"] <= 0
        or settings["MIN_ADAPTIVE_LR"] > settings["LORA_LR"]
    ):
        raise ValueError(
            "the adaptive learning-rate floor must be positive and no greater "
            "than LoRA learning rate"
        )
    if settings["SWEEP_ROWS"] < 0:
        raise ValueError("the checkpoint sweep row count cannot be negative")
    if 3 * settings["MIN_SCORE_BAND_ROWS"] > settings["MIN_TRAIN_ROWS"]:
        raise ValueError("train row minimum must cover the three score-band minimums")
    if (
        2 * settings["MIN_VALIDATION_SCORE_BAND_ROWS"]
        + settings["MIN_VALIDATION_MIDDLE_SCORE_BAND_ROWS"]
        > settings["MIN_VALIDATION_ROWS"]
    ):
        raise ValueError(
            "validation row minimum must cover configured score-band minimums"
        )
    if (
        2 * settings["MIN_GENERATED_TEST_SCORE_BAND_ROWS"]
        + settings["MIN_GENERATED_TEST_MIDDLE_SCORE_BAND_ROWS"]
        > settings["MIN_GENERATED_TEST_ROWS"]
    ):
        raise ValueError("test row minimum must cover configured score-band minimums")
    if (
        settings["ASSUMED_SECONDS_PER_STEP"] <= 0
        or settings["BUDGET_SAFETY_FACTOR"] < 1
    ):
        raise ValueError(
            "training speed estimate must be positive and safety factor at least one"
        )
    if settings["PATIENCE"] < 0:
        raise ValueError("HACKGRADER_PATIENCE cannot be negative")
