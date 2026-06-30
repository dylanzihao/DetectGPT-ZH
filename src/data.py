import sys
import json
import logging
import argparse
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

from config import config  # Must import before huggingface_hub to set HF_ENDPOINT env var

from huggingface_hub import hf_hub_download, list_repo_files

# Configure global logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class QuestionEntry:
    """Represents a question entry containing both human and AI answers"""

    question_id: str  # Unique question identifier
    subset: str  # Subset name this entry belongs to
    human_answers: List[Dict] = field(default_factory=list)  # Human answers, label=0
    ai_answers: List[Dict] = field(default_factory=list)  # AI answers, label=1

    @property
    def total_answers(self) -> int:
        """Total number of answers (human + AI) for this question"""
        return len(self.human_answers) + len(self.ai_answers)

    def flatten(self) -> List[Dict]:
        """Merge all answers into a single list"""
        return self.human_answers + self.ai_answers


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments to override default config"""
    parser = argparse.ArgumentParser(
        description="Process HC3-Chinese dataset for AI text detection"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.paths.processed_data_dir,
        help=f"Output directory (default: {config.paths.processed_data_dir})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.data.seed,
        help=f"Random seed (default: {config.data.seed})",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=config.data.train_ratio,
        help=f"Training set ratio (default: {config.data.train_ratio})",
    )
    parser.add_argument(
        "--dev_ratio",
        type=float,
        default=config.data.dev_ratio,
        help=f"Validation set ratio (default: {config.data.dev_ratio})",
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=config.data.min_text_length,
        help=f"Minimum text length in chars after cleaning (default: {config.data.min_text_length})",
    )
    parser.add_argument(
        "--max_samples_per_subset",
        type=int,
        default=config.data.max_samples_per_subset,
        help="Max question entries per subset (default: no limit)",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        default=config.data.balance,
        help="Balance classes by downsampling the majority class",
    )
    parser.add_argument(
        "--no_question_level_split",
        action="store_true",
        default=not config.data.question_level_split,
        help="Disable question-level split (fall back to record-level)",
    )
    return parser.parse_args()


def setup_reproducibility(seed: int) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Reproducibility configured: seed=%d", seed)


def normalize_text(text: str) -> str:
    """
    Normalize text: strip whitespace, collapse consecutive spaces,
    remove HTML tags, fix extra spaces before punctuation
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+([，。！？、；：,\.!\?;:])", r"\1", text)
    return text.strip()


def clean_text(text: Optional[str]) -> Optional[str]:
    """Clean a single text entry; return None if invalid"""
    if text is None:
        return None
    text = normalize_text(text)
    return text if len(text) > 0 else None


def is_valid_sample(text: str, min_length: int) -> bool:
    """Check if text meets the minimum length requirement"""
    return len(text) >= min_length


def load_hc3_dataset(
    max_samples_per_subset: Optional[int] = None,
) -> Dict[str, List]:
    """
    Load all JSONL subsets of the HC3-Chinese dataset from HuggingFace mirror.

    Args:
        max_samples_per_subset: Maximum question entries per subset

    Returns:
        Dict mapping subset names to lists of question entries
    """
    logger.info(
        "Loading dataset '%s' from mirror %s ...",
        config.paths.dataset_name,
        config.paths.hf_endpoint,
    )

    # List all files in the dataset repository
    try:
        repo_files = list_repo_files(config.paths.dataset_name, repo_type="dataset")
    except Exception as e:
        logger.error("Failed to list dataset files: %s", e)
        sys.exit(1)

    # Filter and sort JSONL files
    jsonl_files = sorted([f for f in repo_files if f.endswith(".jsonl")])
    if not jsonl_files:
        logger.error("No JSONL files found in dataset repository")
        sys.exit(1)

    logger.info("Found %d JSONL subsets: %s", len(jsonl_files), jsonl_files)

    result = {}
    for jsonl_file in jsonl_files:
        config_name = Path(jsonl_file).stem  # Use filename (without extension) as subset name
        logger.info("  Downloading config '%s' (%s) ...", config_name, jsonl_file)
        try:
            # Download JSONL file from HuggingFace
            local_path = hf_hub_download(
                repo_id=config.paths.dataset_name,
                filename=jsonl_file,
                repo_type="dataset",
            )
        except Exception as e:
            logger.error("  Failed to download '%s': %s", jsonl_file, e)
            continue

        # Read JSONL file line by line
        samples = []
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        # Randomly sample if max_samples_per_subset is set and exceeded
        if (
            max_samples_per_subset is not None
            and len(samples) > max_samples_per_subset
        ):
            local_rng = random.Random(config.data.seed)
            local_rng.shuffle(samples)
            samples = samples[:max_samples_per_subset]
            logger.info(
                "  [%s] Limited to %d question entries (randomly sampled)",
                config_name,
                len(samples),
            )
        else:
            logger.info(
                "  [%s] Loaded %d question entries", config_name, len(samples)
            )
        result[config_name] = samples

    return result


def process_subset_question_level(
    subset_name: str,
    samples: List,
    min_text_length: int,
    balance: bool = False,
    seed: int = None,
) -> Tuple[List[QuestionEntry], Dict]:
    """
    Process a single subset at the question level:
    - Extract human answers (label 0) and AI answers (label 1)
    - Clean and filter invalid text
    - Optionally balance classes

    Returns:
        (list of QuestionEntry, statistics dict)
    """
    if seed is None:
        seed = config.data.seed

    total_questions = 0
    total_human_raw = 0
    total_ai_raw = 0
    skipped_human = 0
    skipped_ai = 0
    empty_questions = 0
    question_entries = []

    for entry in samples:
        human_answers = entry.get("human_answers", [])
        chatgpt_answers = entry.get("chatgpt_answers", [])
        question_id = entry.get("id", f"{subset_name}_{total_questions}")

        total_questions += 1
        total_human_raw += len(human_answers)
        total_ai_raw += len(chatgpt_answers)

        valid_human = []
        valid_ai = []

        # Clean and filter human answers
        for answer in human_answers:
            text = clean_text(answer)
            if text is None or not is_valid_sample(text, min_text_length):
                skipped_human += 1
                continue
            valid_human.append({"text": text, "label": 0})

        # Clean and filter AI answers
        for answer in chatgpt_answers:
            text = clean_text(answer)
            if text is None or not is_valid_sample(text, min_text_length):
                skipped_ai += 1
                continue
            valid_ai.append({"text": text, "label": 1})

        # Skip questions where all answers are filtered out
        if not valid_human and not valid_ai:
            empty_questions += 1
            continue

        question_entries.append(
            QuestionEntry(
                question_id=question_id,
                subset=subset_name,
                human_answers=valid_human,
                ai_answers=valid_ai,
            )
        )

    total_valid_human_before = sum(len(q.human_answers) for q in question_entries)
    total_valid_ai_before = sum(len(q.ai_answers) for q in question_entries)

    # Balance classes by downsampling the majority class if enabled
    if balance and total_valid_human_before > 0 and total_valid_ai_before > 0:
        total_human_after, total_ai_after = balance_subset(
            question_entries, seed=seed
        )
        logger.info(
            "  [%s] Balanced: human %d -> %d, chatgpt %d -> %d",
            subset_name,
            total_valid_human_before,
            total_human_after,
            total_valid_ai_before,
            total_ai_after,
        )
    else:
        with_ai = sum(1 for q in question_entries if q.ai_answers)
        with_human = sum(1 for q in question_entries if q.human_answers)
        with_both = sum(
            1 for q in question_entries if q.human_answers and q.ai_answers
        )
        logger.info(
            "  [%s] Questions: %d total, %d with human, %d with AI, %d with both | "
            "human=%d, chatgpt=%d | skipped: human=%d, ai=%d, empty_q=%d",
            subset_name,
            len(question_entries),
            with_human,
            with_ai,
            with_both,
            total_valid_human_before,
            total_valid_ai_before,
            skipped_human,
            skipped_ai,
            empty_questions,
        )

    stats = {
        "total_questions": total_questions,
        "valid_questions": len(question_entries),
        "empty_questions": empty_questions,
        "human_raw": total_human_raw,
        "ai_raw": total_ai_raw,
        "human_valid": total_valid_human_before,
        "ai_valid": total_valid_ai_before,
        "skipped_human": skipped_human,
        "skipped_ai": skipped_ai,
    }

    return question_entries, stats


def balance_subset(
    entries: List[QuestionEntry], seed: int = None
) -> Tuple[int, int]:
    """
    Balance classes in a list of question entries by downsampling the majority class.

    Randomly discards answers from the majority class so that
    human and AI answer counts become equal.

    Returns:
        (total_human_after, total_ai_after)
    """
    if seed is None:
        seed = config.data.seed

    # Collect all human and AI answers
    all_human = []
    all_ai = []
    for q in entries:
        all_human.extend(q.human_answers)
        all_ai.extend(q.ai_answers)

    rng = random.Random(seed)
    # If more human answers, randomly discard some
    if len(all_human) > len(all_ai):
        rng.shuffle(all_human)
        keep_count = len(all_ai)
        discarded = set(id(r) for r in all_human[keep_count:])
        for q in entries:
            q.human_answers = [
                r for r in q.human_answers if id(r) not in discarded
            ]
    # If more AI answers, randomly discard some
    elif len(all_ai) > len(all_human):
        rng.shuffle(all_ai)
        keep_count = len(all_human)
        discarded = set(id(r) for r in all_ai[keep_count:])
        for q in entries:
            q.ai_answers = [
                r for r in q.ai_answers if id(r) not in discarded
            ]

    total_human = sum(len(q.human_answers) for q in entries)
    total_ai = sum(len(q.ai_answers) for q in entries)
    return total_human, total_ai


def split_questions_at_question_level(
    question_entries: List[QuestionEntry],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> Tuple[List[QuestionEntry], List[QuestionEntry], List[QuestionEntry]]:
    """
    Split dataset at the question level: all answers of the same question
    stay in the same split to prevent data leakage.
    """
    rng = random.Random(seed)
    shuffled = question_entries.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    train_questions = shuffled[:train_end]
    dev_questions = shuffled[train_end:dev_end]
    test_questions = shuffled[dev_end:]

    return train_questions, dev_questions, test_questions


def split_records_record_level(
    all_records: List[Dict],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset at the record level: each record is independently
    assigned to train/dev/test. Answers from the same question may
    end up in different splits.
    """
    rng = random.Random(seed)
    shuffled = all_records.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    train_set = shuffled[:train_end]
    dev_set = shuffled[train_end:dev_end]
    test_set = shuffled[dev_end:]

    return train_set, dev_set, test_set


def flatten_questions(questions: List[QuestionEntry]) -> List[Dict]:
    """Flatten question entries into a list of records (each with text and label)"""
    records = []
    for q in questions:
        records.extend(q.flatten())
    return records


def shuffle_records(records: List[Dict], seed: int) -> List[Dict]:
    """Randomly shuffle a list of records"""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    return shuffled


def save_jsonl(records: List[Dict], filepath: Path) -> None:
    """Save records as JSONL format (one JSON object per line)"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), filepath)


def compute_label_distribution(records: List[Dict], name: str) -> None:
    """Compute and log label distribution (human vs AI ratio) for a dataset"""
    labels = [r["label"] for r in records]
    total = len(labels)
    if total == 0:
        logger.warning("  [%s] (empty set)", name)
        return
    human_count = sum(1 for l in labels if l == 0)
    ai_count = sum(1 for l in labels if l == 1)
    logger.info(
        "  [%s] %d samples | human=%.1f%% (%d) | chatgpt=%.1f%% (%d)",
        name,
        total,
        human_count / total * 100,
        human_count,
        ai_count / total * 100,
        ai_count,
    )


def log_subset_stats_summary(all_stats: Dict[str, Dict]) -> None:
    """Log a table summarizing statistics for all subsets"""
    headers = [
        "Subset",
        "Questions",
        "Human(raw)",
        "AI(raw)",
        "Human(valid)",
        "AI(valid)",
        "Skipped(H)",
        "Skipped(A)",
    ]
    sep = " | "
    rule = "-" * 90

    logger.info("Per-subset processing summary:")
    logger.info(rule)
    logger.info(sep.join([f"{h:>14}" for h in headers]))
    logger.info(rule)

    (
        total_q,
        total_h_raw,
        total_a_raw,
        total_h_val,
        total_a_val,
        total_sk_h,
        total_sk_a,
    ) = (0, 0, 0, 0, 0, 0, 0)
    for name, s in all_stats.items():
        logger.info(
            sep.join(
                [
                    f"{name:>14}",
                    f"{s['valid_questions']:>9d}",
                    f"{s['human_raw']:>11d}",
                    f"{s['ai_raw']:>9d}",
                    f"{s['human_valid']:>12d}",
                    f"{s['ai_valid']:>10d}",
                    f"{s['skipped_human']:>10d}",
                    f"{s['skipped_ai']:>10d}",
                ]
            )
        )
        total_q += s["valid_questions"]
        total_h_raw += s["human_raw"]
        total_a_raw += s["ai_raw"]
        total_h_val += s["human_valid"]
        total_a_val += s["ai_valid"]
        total_sk_h += s["skipped_human"]
        total_sk_a += s["skipped_ai"]

    logger.info(rule)
    logger.info(
        sep.join(
            [
                f"{'TOTAL':>14}",
                f"{total_q:>9d}",
                f"{total_h_raw:>11d}",
                f"{total_a_raw:>9d}",
                f"{total_h_val:>12d}",
                f"{total_a_val:>10d}",
                f"{total_sk_h:>10d}",
                f"{total_sk_a:>10d}",
            ]
        )
    )
    logger.info(rule)


def main() -> None:
    """Main pipeline: download data -> process subsets -> split dataset -> save as JSONL"""
    args = parse_args()
    setup_reproducibility(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir.resolve())

    # Load raw data
    raw_data = load_hc3_dataset(max_samples_per_subset=args.max_samples_per_subset)

    all_question_entries = []
    all_subset_stats = {}

    # Process each subset
    logger.info("Processing subsets at question level ...")
    for subset_name, samples in raw_data.items():
        question_entries, stats = process_subset_question_level(
            subset_name=subset_name,
            samples=samples,
            min_text_length=args.min_text_length,
            balance=args.balance,
            seed=args.seed,
        )
        all_question_entries.extend(question_entries)
        all_subset_stats[subset_name] = stats

    log_subset_stats_summary(all_subset_stats)

    # Compute aggregate statistics
    total_questions = len(all_question_entries)
    total_records = sum(q.total_answers for q in all_question_entries)
    total_human = sum(len(q.human_answers) for q in all_question_entries)
    total_ai = sum(len(q.ai_answers) for q in all_question_entries)
    logger.info(
        "Aggregate: %d questions, %d total samples (human=%d, chatgpt=%d)",
        total_questions,
        total_records,
        total_human,
        total_ai,
    )

    if total_records == 0:
        logger.error("No valid samples after processing. Exiting.")
        sys.exit(1)

    # Normalize ratios if they exceed 1.0
    split_ratio_sum = args.train_ratio + args.dev_ratio
    if split_ratio_sum > 1.0:
        logger.warning(
            "train_ratio + dev_ratio = %.2f > 1.0, normalizing ...",
            split_ratio_sum,
        )
        args.train_ratio = args.train_ratio / split_ratio_sum
        args.dev_ratio = args.dev_ratio / split_ratio_sum

    use_question_level = not args.no_question_level_split
    logger.info(
        "Splitting strategy: %s-level (train=%.0f%%, dev=%.0f%%, test=%.0f%%)",
        "question" if use_question_level else "record",
        args.train_ratio * 100,
        args.dev_ratio * 100,
        (1 - args.train_ratio - args.dev_ratio) * 100,
    )

    # Split according to the selected strategy
    if use_question_level:
        train_q, dev_q, test_q = split_questions_at_question_level(
            all_question_entries,
            args.train_ratio,
            args.dev_ratio,
            args.seed,
        )
        logger.info(
            "Questions per split: train=%d, dev=%d, test=%d",
            len(train_q),
            len(dev_q),
            len(test_q),
        )
        train_set = shuffle_records(flatten_questions(train_q), args.seed)
        dev_set = shuffle_records(flatten_questions(dev_q), args.seed + 1)
        test_set = shuffle_records(flatten_questions(test_q), args.seed + 2)
    else:
        all_records = flatten_questions(all_question_entries)
        train_set, dev_set, test_set = split_records_record_level(
            all_records,
            args.train_ratio,
            args.dev_ratio,
            args.seed,
        )

    # Log label distributions for each split
    logger.info("=== Label Distribution ===")
    compute_label_distribution(train_set, "train")
    compute_label_distribution(dev_set, "dev")
    compute_label_distribution(test_set, "test")

    # Save as JSONL files
    save_jsonl(train_set, output_dir / config.paths.train_file_name)
    save_jsonl(dev_set, output_dir / config.paths.dev_file_name)
    save_jsonl(test_set, output_dir / config.paths.test_file_name)

    logger.info("All done! Processed files are in: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
