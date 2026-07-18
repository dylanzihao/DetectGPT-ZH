import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

logger = logging.getLogger(__name__)


def get_dtype(cfg) -> tuple:
    """
    Auto-detect GPU capability and return the best precision settings.

    Priority: user explicit config > GPU supports bf16 > fp16 > fp32.

    Returns:
        (torch_dtype, fp16_enabled, bf16_enabled) tuple
    """
    if cfg.hardware.fp16:
        return torch.float16, True, False
    if cfg.hardware.bf16:
        return torch.bfloat16, False, True
    if not torch.cuda.is_available():
        return torch.float32, False, False

    # Check if GPU supports bfloat16
    try:
        bf16_supported = torch.cuda.is_bf16_supported()
    except AttributeError:
        bf16_supported = False

    if bf16_supported:
        logger.info("GPU supports bfloat16, loading model in bf16")
        return torch.bfloat16, False, True

    compute_capability = torch.cuda.get_device_capability()
    logger.info(
        "GPU does not support bfloat16 (compute %s), loading model in fp16",
        compute_capability,
    )
    return torch.float16, True, False


def setup_data_paths(cfg) -> Dict[str, str]:
    """
    Auto-discover train/validation/test data file paths from config.

    Lookup order:
    1. Explicitly specified train_file/dev_file/test_file
    2. Search for train.jsonl/train.json etc. in data_dir

    Returns:
        Dict mapping split names to file paths
    """
    if cfg.paths.train_file and cfg.paths.dev_file:
        paths = {
            "train": cfg.paths.train_file,
            "dev": cfg.paths.dev_file,
        }
        if cfg.paths.test_file:
            paths["test"] = cfg.paths.test_file
        return paths

    if cfg.paths.data_dir:
        data_dir = Path(cfg.paths.data_dir)
        paths = {}
        for split in ["train", "dev", "test"]:
            for ext in [".jsonl", ".json"]:
                candidate = data_dir / f"{split}{ext}"
                if candidate.exists():
                    paths[split] = str(candidate)
                    break
        return paths

    raise FileNotFoundError(
        "No data found. Set data_dir or train_file/dev_file in config.py"
    )


def load_and_prepare_data(
    data_paths: dict,
    tokenizer,
    max_length: int,
    max_train_samples: int = 0,
) -> DatasetDict:
    """
    Load JSONL data and perform tokenization preprocessing.

    Pipeline:
    1. Load data splits from JSONL files
    2. Optionally limit training samples
    3. Tokenize text (padding=False for dynamic padding)
    4. Rename label column, remove text column, convert to PyTorch tensors
    5. If no validation set, split 10% from training set

    Returns:
        DatasetDict with train/validation/test splits
    """
    dataset_dict = {}

    # Load each data split
    for split, filepath in data_paths.items():
        logger.info("Loading %s data from %s ...", split, filepath)
        dataset = load_dataset("json", data_files=filepath, split="train")

        if (
            split == "train"
            and max_train_samples > 0
            and len(dataset) > max_train_samples
        ):
            dataset = dataset.select(range(max_train_samples))
            logger.info("  %s samples (subset): %d", split, len(dataset))
        else:
            logger.info("  %s samples: %d", split, len(dataset))

        dataset_dict[split] = dataset

    if "train" not in dataset_dict:
        raise ValueError("At least 'train' split is required")

    def tokenize_function(examples):
        """Tokenize a batch of text with truncation"""
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    # Pipeline process each split
    tokenized_datasets = {}
    for split, dataset in dataset_dict.items():
        tokenized = dataset.map(
            tokenize_function, batched=True, desc=f"Tokenizing {split}"
        )
        tokenized = tokenized.rename_column("label", "labels")  # Align with Transformers default column name
        tokenized = tokenized.remove_columns(["text"])  # Keep only input_ids/attention_mask
        tokenized = tokenized.with_format("torch")  # Convert to PyTorch tensors
        tokenized_datasets[split] = tokenized

    if "dev" in tokenized_datasets:
        result = DatasetDict({
            "train": tokenized_datasets["train"],
            "validation": tokenized_datasets["dev"],
        })
    else:
        # Split 10% from training set for validation if no dev set provided
        logger.info("No dev split found, splitting train set (test_size=0.1)")
        split_ds = tokenized_datasets["train"].train_test_split(
            test_size=0.1, seed=42
        )
        result = DatasetDict({
            "train": split_ds["train"],
            "validation": split_ds["test"],
        })

    if "test" in tokenized_datasets:
        result["test"] = tokenized_datasets["test"]

    return result


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute classification evaluation metrics.

    Metrics:
    - accuracy: Overall accuracy
    - f1: F1 score (primary metric)
    - precision: Precision
    - recall: Recall
    - specificity: True negative rate

    Returns:
        Dict mapping metric names to values
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    precision = precision_score(labels, preds, average="binary", zero_division=0)
    recall = recall_score(labels, preds, average="binary", zero_division=0)

    # Calculate specificity from confusion matrix: TN / (TN + FP)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def setup_logging(log_dir: str, prefix: str = "train") -> None:
    """
    Configure logging system: output to both console and file.

    Log files are saved in log_dir with timestamps in the filename.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{prefix}_{timestamp}.log"

    # Add file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    _ensure_library_loggers_propagate()

    root_logger.info("Log file: %s", log_file)


_LIBRARY_LOGGER_PREFIXES = ("transformers", "datasets")


def _ensure_library_loggers_propagate() -> None:
    """
    Ensure third-party library logs propagate to the root logger,
    preventing them from being suppressed by their own handlers.
    """
    for name in logging.root.manager.loggerDict:
        for prefix in _LIBRARY_LOGGER_PREFIXES:
            if name.startswith(prefix):
                log = logging.getLogger(name)
                log.setLevel(logging.INFO)
                log.propagate = True
    for prefix in _LIBRARY_LOGGER_PREFIXES:
        logging.getLogger(prefix).propagate = True
