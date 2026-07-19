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
    # --- C-ReD dataset (本地) ---
    cred_local_dir: str = r"D:\Dylan\Data\C-ReD\benchmark data"  # C-ReD 本地路径
    cred_domains: tuple = ("news", "question answer", "film review", "paper", "composition")  # C-ReD 领域
    data_dir: str = "/kaggle/input/datasets/dylanzihao/detectgpt-training-dataset"  # 预处理后数据目录
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
    # --- 文本过滤 ---
    min_text_length: int = 20  # 最小有效文本长度（字符）
    max_text_length: int = 4096  # 最大有效文本长度（字符），超过截断
    min_chinese_ratio: float = 0.3  # 最小中文字符占比（过滤纯英文/数字文本）
    # --- 文本清洗 ---
    clean_control_chars: bool = True  # 移除控制字符
    clean_urls: bool = True  # 规范化 URL（替换为 <URL> 占位符）
    normalize_unicode: bool = True  # Unicode NFC 规范化
    # --- 去重 ---
    deduplicate: bool = True  # 是否去除重复文本
    dedup_method: str = "exact"  # 去重方式: "exact"(完全匹配) | "normalized"(清洗后匹配)
    # --- 类别平衡 ---
    balance: bool = True  # 是否平衡类别
    balance_strategy: str = "augment"  # 平衡策略: "downsample"(下采样多数类) | "upsample"(上采样少数类) | "augment"(nlpcda 增强少数类)
    # --- nlpcda 文本增强 (仅用于 augment 策略) ---
    augment_target_count: int = 30000  # 增强目标：少数类增强后的总数
    augment_change_rate: float = 0.15  # 增强时的文本改变率 (0.0~1.0)
    augment_methods: tuple = (
        "synonym",       # 近义词替换 (Similarword)
        "char_delete",   # 随机字删除 (RandomDeleteChar)
        "char_swap",     # 邻近字换位 (CharPositionExchange)
    )
    # --- 分割策略 ---
    stratified_split: bool = True  # 是否按领域分层分割（保证各领域均匀分布）
    question_level_split: bool = True  # (已废弃，C-ReD 为记录级数据)
    max_train_samples: int = 0  # Max training samples (0 = full dataset)
    max_samples_per_subset: Optional[int] = None  # Max question entries per subset, None = unlimited


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
