import os
import json
import logging

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

from config import config
from utils import (
    get_dtype,
    setup_data_paths,
    load_and_prepare_data,
    compute_metrics,
    setup_logging,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(cfg, torch_dtype):
    """
    Load pretrained model and tokenizer, then configure LoRA.

    Steps:
    1. Load tokenizer, handle missing pad_token
    2. Load sequence classification model (with label mapping)
    3. Configure LoRA parameters and wrap the model

    Returns:
        (peft_model, tokenizer)
    """
    logger.info("Loading tokenizer: %s ...", cfg.paths.model_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.model_path)

    # Fall back to eos_token if pad_token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token: %s", tokenizer.pad_token)

    logger.info("Loading model: %s ...", cfg.paths.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.paths.model_path,
        num_labels=cfg.model.num_labels,
        id2label=cfg.model.id2label,
        label2id=cfg.model.label2id,
        dtype=torch_dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Configure LoRA for parameter-efficient fine-tuning
    logger.info(
        "Configuring LoRA (r=%d, alpha=%d) ...",
        cfg.training.lora_r,
        cfg.training.lora_alpha,
    )
    lora_config = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        target_modules=cfg.model.lora_target_modules,
        lora_dropout=cfg.training.lora_dropout,
        bias=cfg.model.lora_bias,
        task_type=TaskType[cfg.model.lora_task_type],
    )

    # Wrap the base model with LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(cfg):
    """
    Main training pipeline.

    Steps:
    1. Set random seed and create directories
    2. Detect GPU precision and load model
    3. Load and tokenize data
    4. Configure training arguments (lr, batch size, early stopping, etc.)
    5. Train using the Trainer API
    6. Save model and evaluate on test set
    """
    set_seed(cfg.training.seed)

    # Auto-detect Kaggle environment and switch output directory
    if cfg.paths.output_dir == "./output" and os.path.exists(
        cfg.paths.kaggle_working_dir
    ):
        cfg.paths.output_dir = cfg.paths.kaggle_working_dir + "/model"

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    setup_logging(cfg.paths.log_dir)

    # Auto-detect best GPU precision
    torch_dtype, fp16_enabled, bf16_enabled = get_dtype(cfg)

    model, tokenizer = setup_model_and_tokenizer(cfg, torch_dtype)

    # Load data paths and perform tokenization
    data_paths = setup_data_paths(cfg)
    logger.info("Data paths: %s", data_paths)
    tokenized_datasets = load_and_prepare_data(
        data_paths, tokenizer, cfg.training.max_length, cfg.data.max_train_samples
    )

    # Calculate warmup steps: total training steps * warmup ratio
    total_train_steps = (
        len(tokenized_datasets["train"])
        + cfg.training.batch_size * cfg.training.gradient_accumulation_steps
        - 1
    ) // (
        cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    ) * cfg.training.num_epochs
    warmup_steps = int(total_train_steps * cfg.training.warmup_ratio)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=cfg.paths.output_dir,
        eval_strategy=cfg.training.eval_strategy,
        save_strategy=cfg.training.save_strategy,
        logging_strategy=cfg.training.logging_strategy,
        logging_steps=cfg.training.logging_steps,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_epochs,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,  # Load the best checkpoint when training finishes
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        save_total_limit=cfg.training.save_total_limit,
        seed=cfg.training.seed,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
        ddp_find_unused_parameters=cfg.training.ddp_find_unused_parameters,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        optim=cfg.training.optimizer,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        report_to=cfg.training.report_to,
    )

    # Dynamic padding data collator: pad to the longest sequence in each batch
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=cfg.training.padding,
    )

    # Create Trainer with early stopping callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience
            )
        ],
    )

    # Start training
    logger.info("Starting training ...")
    trainer.train()

    # Save model and tokenizer
    logger.info("Saving model to %s ...", cfg.paths.output_dir)
    trainer.save_model(cfg.paths.output_dir)
    tokenizer.save_pretrained(cfg.paths.output_dir)

    # Evaluate on test set
    if "test" in tokenized_datasets:
        logger.info("Evaluating on test set ...")
        test_results = trainer.evaluate(tokenized_datasets["test"])
        logger.info("Test results: %s", test_results)

        results_path = os.path.join(
            cfg.paths.output_dir, cfg.paths.test_results_file
        )
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)

    logger.info("Training complete! Model saved to: %s", cfg.paths.output_dir)


if __name__ == "__main__":
    train(config)
