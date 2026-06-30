import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# Set HuggingFace mirror for faster downloads in China, disable symlink warnings
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TQDM_MININTERVAL"] = "30"


@dataclass
class PathsConfig:
    """Paths configuration: data, model, output directories, etc."""

    hf_endpoint: str = "https://hf-mirror.com"
    dataset_name: str = "Hello-SimpleAI/HC3-Chinese"  # Dataset name on HuggingFace
    data_dir: Optional[str] = "/kaggle/input/datasets/dylanzihao/modified-hc3-chinese-for-transformers-training"  # Preprocessed data directory
    train_file: Optional[str] = None  # Explicit train file path (takes precedence over data_dir)
    dev_file: Optional[str] = None  # Explicit validation file path
    test_file: Optional[str] = None  # Explicit test file path
    processed_data_dir: str = field(
        default_factory=lambda: str(Path(__file__).resolve().parent.parent / "data")
    )  # Default directory for processed data
    model_path: str = "/kaggle/input/models/dylanzihao/chinese-bert-wwm-ext/transformers/default/1"  # Pretrained model path
    output_dir: str = "/kaggle/working/model"  # Model output directory
    log_dir: str = "/kaggle/working/logs/"  # Log directory
    kaggle_working_dir: str = "/kaggle/working"  # Kaggle working directory
    train_file_name: str = "train.jsonl"  # Output train filename
    dev_file_name: str = "dev.jsonl"  # Output dev filename
    test_file_name: str = "test.jsonl"  # Output test filename
    test_results_file: str = "test_results.json"  # Test results filename


@dataclass
class DataConfig:
    """Data processing configuration: random seed, split ratios, text filtering, etc."""

    seed: int = 42  # Random seed for reproducibility
    train_ratio: float = 0.8  # Training set ratio
    dev_ratio: float = 0.1  # Validation set ratio
    test_ratio: float = 0.1  # Test set ratio
    min_text_length: int = 5  # Minimum valid text length (characters)
    max_samples_per_subset: Optional[int] = None  # Max question entries per subset, None = unlimited
    balance: bool = True  # Whether to balance classes (downsample majority)
    question_level_split: bool = True  # Whether to split at question level to prevent data leakage
    max_train_samples: int = 0  # Max training samples (0 = full dataset)


@dataclass
class ModelConfig:
    """Model configuration: classification head, LoRA target modules, etc."""

    num_labels: int = 2  # Number of classes: 0 (human), 1 (AI)
    id2label: Dict[int, str] = field(default_factory=lambda: {0: "human", 1: "ai"})
    label2id: Dict[str, int] = field(default_factory=lambda: {"human": 0, "ai": 1})
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )  # LoRA target modules
    lora_bias: str = "none"  # Whether to train bias parameters in LoRA
    lora_task_type: str = "SEQ_CLS"  # LoRA task type: sequence classification


@dataclass
class TrainingConfig:
    """Training hyperparameters: batch size, learning rate, LoRA params, early stopping, etc."""

    max_length: int = 512  # Maximum sequence length (tokens)
    batch_size: int = 16  # Training batch size per device
    eval_batch_size: int = 16  # Evaluation batch size per device
    gradient_accumulation_steps: int = 2  # Gradient accumulation steps (effective batch size multiplier)
    learning_rate: float = 2e-5  # Initial learning rate
    num_epochs: int = 5  # Number of training epochs
    weight_decay: float = 0.01  # Weight decay coefficient (L2 regularization)
    warmup_ratio: float = 0.1  # Ratio of total steps used for learning rate warmup
    logging_steps: int = 100  # Log every N steps
    save_total_limit: int = 2  # Maximum number of checkpoints to keep
    seed: int = 42  # Training random seed
    eval_strategy: str = "epoch"  # Evaluation strategy: evaluate at each epoch end
    save_strategy: str = "epoch"  # Save strategy: save at each epoch end
    logging_strategy: str = "steps"  # Logging strategy: log by steps
    metric_for_best_model: str = "f1"  # Metric to select the best model
    greater_is_better: bool = True  # Whether the metric is higher-is-better
    early_stopping_patience: int = 3  # Early stopping patience: stop if metric doesn't improve for N evals
    dataloader_num_workers: int = 4  # Dataloader worker processes
    dataloader_pin_memory: bool = True  # Whether to use pinned memory for faster data transfer
    ddp_find_unused_parameters: bool = False  # Whether to find unused parameters in DDP
    optimizer: str = "adamw_torch"  # Optimizer type
    lr_scheduler_type: str = "cosine"  # Learning rate scheduler: cosine annealing
    report_to: str = "none"  # Reporting target (none = don't report externally)
    padding: str = "longest"  # Padding strategy: pad to the longest sequence in each batch
    disable_tqdm: bool = False  # Whether to disable tqdm progress bar
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA scaling parameter alpha
    lora_dropout: float = 0.05  # LoRA dropout rate


@dataclass
class HardwareConfig:
    """Hardware precision configuration: enable fp16 or bf16 mixed precision training"""

    fp16: bool = False
    bf16: bool = False


@dataclass
class AppConfig:
    """Top-level configuration class aggregating all config groups"""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


# Global singleton config object
config = AppConfig()
