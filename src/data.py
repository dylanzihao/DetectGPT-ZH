import sys
import json
import csv
import logging
import argparse
import re
import random
import unicodedata
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TextRecord:
    """A single text record with label: 0=human, 1=AI."""
    __slots__ = ("text", "label", "domain", "generator")

    def __init__(self, text: str, label: int, domain: str = "", generator: str = ""):
        self.text = text
        self.label = label
        self.domain = domain
        self.generator = generator

    @property
    def source(self) -> str:
        return f"C-ReD/{self.domain}/{self.generator}"

    @property
    def text_hash(self) -> str:
        """SHA256 hash of normalized text, used for deduplication."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def text_length(self) -> int:
        return len(self.text)

    @property
    def chinese_char_count(self) -> int:
        return sum(1 for c in self.text if '\u4e00' <= c <= '\u9fff')

    @property
    def chinese_ratio(self) -> float:
        return self.chinese_char_count / max(self.text_length, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="处理 C-ReD 数据集，生成训练/验证/测试 JSONL"
    )
    parser.add_argument("--output_dir", type=str, default=config.paths.processed_data_dir,
                        help=f"输出目录 (默认: {config.paths.processed_data_dir})")
    parser.add_argument("--seed", type=int, default=config.data.seed)
    parser.add_argument("--train_ratio", type=float, default=config.data.train_ratio)
    parser.add_argument("--dev_ratio", type=float, default=config.data.dev_ratio)
    parser.add_argument("--min_text_length", type=int, default=config.data.min_text_length)
    parser.add_argument("--max_text_length", type=int, default=config.data.max_text_length)
    parser.add_argument("--min_chinese_ratio", type=float, default=config.data.min_chinese_ratio)
    parser.add_argument("--no_dedup", action="store_true",
                        help="禁用文本去重")
    parser.add_argument("--balance", action="store_true", default=config.data.balance)
    parser.add_argument("--balance_strategy", type=str, default=config.data.balance_strategy,
                        choices=("downsample", "upsample"),
                        help="平衡策略: downsample(下采样) | upsample(上采样)")
    parser.add_argument("--no_stratified", action="store_true",
                        help="禁用分层分割，使用纯随机分割")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Text cleaning pipeline
# ---------------------------------------------------------------------------

# URL pattern (loose match)
_URL_RE = re.compile(r'https?://\S+|www\.\S+')
# Control characters (except newlines and tabs)
_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
# HTML tags
_HTML_RE = re.compile(r'<[^>]+>')
# Multiple whitespace
_WHITESPACE_RE = re.compile(r'\s+')
# Whitespace before CJK punctuation
_SPACE_BEFORE_CJK_PUNC_RE = re.compile(r'\s+([，。！？、；：])')


def clean_text(text: str, min_len: int = 0) -> Optional[str]:
    """
    Multi-stage text cleaning pipeline.

    Stages:
      1. Strip + normalize whitespace
      2. Remove control characters
      3. Remove HTML tags
      4. Normalize URLs to placeholder
      5. Unicode NFC normalization
      6. Fix spaces before CJK punctuation
      7. Collapse consecutive whitespace
      8. Length truncation
      9. Strip + min length check
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    # Collapse whitespace first
    text = _WHITESPACE_RE.sub(' ', text)

    # Remove control characters
    if config.data.clean_control_chars:
        text = _CONTROL_RE.sub('', text)

    # Remove HTML tags (keep text content)
    text = _HTML_RE.sub('', text)

    # Normalize URLs to placeholder
    if config.data.clean_urls:
        text = _URL_RE.sub('<URL>', text)

    # Unicode NFC normalization
    if config.data.normalize_unicode:
        text = unicodedata.normalize('NFC', text)

    # Fix whitespace before CJK punctuation
    text = _SPACE_BEFORE_CJK_PUNC_RE.sub(r'\1', text)

    # Collapse whitespace again
    text = _WHITESPACE_RE.sub(' ', text)

    # Truncate if too long
    max_len = config.data.max_text_length
    if len(text) > max_len:
        text = text[:max_len]

    text = text.strip()
    if min_len > 0 and len(text) < min_len:
        return None
    return text


def is_valid_record(rec: TextRecord) -> bool:
    """Filter out invalid records based on length and Chinese ratio."""
    if rec.text_length < config.data.min_text_length:
        return False
    if rec.text_length > config.data.max_text_length:
        return False
    if rec.chinese_ratio < config.data.min_chinese_ratio:
        return False
    return True


# ---------------------------------------------------------------------------
# C-ReD loader
# ---------------------------------------------------------------------------

def load_cred_dataset() -> Tuple[List[TextRecord], Dict]:
    """
    Load C-ReD CSV files from local directory with cleaning and filtering.

    C-ReD label mapping: label=1=human, label=0=AI  →  our: label=0=human, 1=AI

    Returns:
        (records, stats) where stats includes per-domain and filter breakdowns
    """
    cred_dir = Path(config.paths.cred_local_dir)
    records = []
    stats = {
        "raw_loaded": 0,
        "cleaned_invalid": 0,
        "filtered_short": 0,
        "filtered_long": 0,
        "filtered_lang": 0,
        "by_domain": defaultdict(lambda: {"human": 0, "ai": 0}),
    }

    for domain in config.paths.cred_domains:
        domain_path = cred_dir / domain
        if not domain_path.exists():
            logger.warning("  领域目录不存在，跳过: %s", domain_path)
            continue

        csv_files = sorted(domain_path.glob("*.csv"))
        for csv_file in csv_files:
            generator = csv_file.stem.replace(
                f"CReD_{domain.replace(' ', '_')}_", ""
            )
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        stats["raw_loaded"] += 1

                        raw_text = row.get("text", "")
                        cleaned = clean_text(raw_text)
                        if cleaned is None:
                            stats["cleaned_invalid"] += 1
                            continue

                        cred_label = int(row.get("label", -1))
                        if cred_label not in (0, 1):
                            continue
                        our_label = 1 - cred_label  # flip

                        rec = TextRecord(
                            text=cleaned, label=our_label,
                            domain=domain, generator=generator,
                        )

                        if not is_valid_record(rec):
                            if rec.text_length < config.data.min_text_length:
                                stats["filtered_short"] += 1
                            elif rec.text_length > config.data.max_text_length:
                                stats["filtered_long"] += 1
                            else:
                                stats["filtered_lang"] += 1
                            continue

                        records.append(rec)
                        if our_label == 0:
                            stats["by_domain"][domain]["human"] += 1
                        else:
                            stats["by_domain"][domain]["ai"] += 1

            except Exception as e:
                logger.error("  读取失败 %s: %s", csv_file, e)
                continue

    return records, stats


def log_load_statistics(stats: Dict) -> None:
    """Log detailed loading statistics."""
    logger.info("=" * 60)
    logger.info("加载统计:")
    logger.info("  原始读取:  %d 条", stats["raw_loaded"])
    logger.info("  清洗丢弃:  %d 条 (空白/无效)", stats["cleaned_invalid"])
    logger.info("  过短过滤:  %d 条 (< min_text_length)", stats["filtered_short"])
    logger.info("  过长截断:  %d 条 (> max_text_length)", stats["filtered_long"])
    logger.info("  语言过滤:  %d 条 (中文字符占比不足)", stats["filtered_lang"])

    logger.info("  各领域分布:")
    total_domain = {"human": 0, "ai": 0}
    for domain in config.paths.cred_domains:
        d = stats["by_domain"].get(domain, {"human": 0, "ai": 0})
        logger.info("    %-20s human=%-6d AI=%-6d", domain, d["human"], d["ai"])
        total_domain["human"] += d["human"]
        total_domain["ai"] += d["ai"]
    logger.info("    %-20s human=%-6d AI=%-6d", "TOTAL", total_domain["human"], total_domain["ai"])


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_records(records: List[TextRecord], method: str) -> Tuple[List[TextRecord], int]:
    """
    Remove duplicate texts.

    Args:
        method: "exact" for exact hash match, "normalized" for cleaned text match

    Returns:
        (deduplicated records, number of duplicates removed)
    """
    seen = set()
    unique = []
    dupes = 0

    for rec in records:
        key = rec.text_hash if method == "exact" else rec.text
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
            unique.append(rec)

    return unique, dupes


# ---------------------------------------------------------------------------
# Class balancing
# ---------------------------------------------------------------------------

def balance_records(
    records: List[TextRecord], strategy: str, seed: int
) -> List[TextRecord]:
    """
    Balance class distribution.

    Args:
        strategy: "downsample" (discard majority) | "upsample" (duplicate minority)
    """
    human_recs = [r for r in records if r.label == 0]
    ai_recs = [r for r in records if r.label == 1]

    n_human, n_ai = len(human_recs), len(ai_recs)
    if n_human == 0 or n_ai == 0:
        return records
    if n_human == n_ai:
        logger.info("  类别已平衡 (human=%d, AI=%d)，无需调整", n_human, n_ai)
        return records

    rng = random.Random(seed)

    if strategy == "downsample":
        if n_human > n_ai:
            rng.shuffle(human_recs)
            human_recs = human_recs[:n_ai]
        else:
            rng.shuffle(ai_recs)
            ai_recs = ai_recs[:n_human]
        logger.info("  下采样: human %d→%d, AI %d→%d",
                     n_human, len(human_recs), n_ai, len(ai_recs))

    elif strategy == "upsample":
        target = max(n_human, n_ai)
        if n_human < target:
            extra = rng.choices(human_recs, k=target - n_human)
            human_recs.extend(extra)
        else:
            extra = rng.choices(ai_recs, k=target - n_ai)
            ai_recs.extend(extra)
        logger.info("  上采样: human %d→%d, AI %d→%d",
                     n_human, len(human_recs), n_ai, len(ai_recs))

    balanced = human_recs + ai_recs
    rng.shuffle(balanced)
    return balanced


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    records: List[TextRecord],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> Tuple[List[TextRecord], List[TextRecord], List[TextRecord]]:
    """
    Split records with stratification by domain (and label within domain),
    ensuring each split has proportional representation of all domains.

    Algorithm:
      1. Group records by (domain, label)
      2. Within each group, shuffle and split by ratios
      3. Merge splits across all groups
      4. Shuffle each final split
    """
    rng = random.Random(seed)

    # Group by (domain, label)
    groups: Dict[Tuple[str, int], List[TextRecord]] = defaultdict(list)
    for rec in records:
        groups[(rec.domain, rec.label)].append(rec)

    train_set, dev_set, test_set = [], [], []

    for (domain, label), group_recs in groups.items():
        rng.shuffle(group_recs)
        n = len(group_recs)
        train_end = int(n * train_ratio)
        dev_end = train_end + int(n * dev_ratio)

        train_set.extend(group_recs[:train_end])
        dev_set.extend(group_recs[train_end:dev_end])
        test_set.extend(group_recs[dev_end:])

    # Shuffle each final split
    rng.shuffle(train_set)
    rng.shuffle(dev_set)
    rng.shuffle(test_set)

    return train_set, dev_set, test_set


def simple_split(
    records: List[TextRecord],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> Tuple[List[TextRecord], List[TextRecord], List[TextRecord]]:
    """Simple random shuffle and split (no stratification)."""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    return shuffled[:train_end], shuffled[train_end:dev_end], shuffled[dev_end:]


# ---------------------------------------------------------------------------
# Statistics & reporting
# ---------------------------------------------------------------------------

def compute_statistics(records: List[TextRecord], name: str) -> Dict:
    """Compute comprehensive statistics for a set of records."""
    if not records:
        return {}

    lengths = [r.text_length for r in records]
    labels = [r.label for r in records]
    human_count = sum(1 for l in labels if l == 0)
    ai_count = sum(1 for l in labels if l == 1)

    # Per-domain distribution
    domain_dist: Dict[str, Dict[str, int]] = defaultdict(lambda: {"human": 0, "ai": 0})
    for r in records:
        if r.label == 0:
            domain_dist[r.domain]["human"] += 1
        else:
            domain_dist[r.domain]["ai"] += 1

    return {
        "name": name,
        "total": len(records),
        "human": human_count,
        "ai": ai_count,
        "human_pct": human_count / len(records) * 100,
        "ai_pct": ai_count / len(records) * 100,
        "avg_length": np.mean(lengths),
        "median_length": np.median(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "std_length": np.std(lengths),
        "domain_dist": dict(domain_dist),
    }


def log_split_statistics(stats_list: List[Dict]) -> None:
    """Log detailed statistics for train/dev/test splits."""
    logger.info("=" * 60)
    logger.info("=== 数据集统计 ===")

    header = f"{'Split':<8} {'总数':>8} {'Human':>8} {'AI':>8} {'AvgLen':>8} {'MedianLen':>8} {'Min':>6} {'Max':>6}"
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)

    for s in stats_list:
        logger.info(
            f"{s['name']:<8} {s['total']:>8} {s['human']:>8} {s['ai']:>8} "
            f"{s['avg_length']:>8.0f} {s['median_length']:>8.0f} "
            f"{s['min_length']:>6} {s['max_length']:>6}"
        )
    logger.info(sep)

    # Per-domain breakdown for each split
    for s in stats_list:
        logger.info("  [%s] 领域分布:", s["name"])
        for domain in sorted(s["domain_dist"]):
            dd = s["domain_dist"][domain]
            total_d = dd["human"] + dd["ai"]
            logger.info("    %-20s total=%-6d human=%-5d AI=%-5d",
                         domain, total_d, dd["human"], dd["ai"])


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_jsonl(records: List[TextRecord], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"text": r.text, "label": r.label}, ensure_ascii=False) + "\n")
    logger.info("保存 %d 条记录 → %s", len(records), filepath)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("输出目录: %s", output_dir.resolve())

    # ---- 1. Load & clean ----
    logger.info("=" * 60)
    logger.info("[1/6] 加载并清洗 C-ReD 数据集 ...")
    records, load_stats = load_cred_dataset()
    log_load_statistics(load_stats)
    logger.info("清洗后有效记录: %d 条", len(records))

    if not records:
        logger.error("没有有效数据，退出。")
        sys.exit(1)

    # ---- 2. Deduplicate ----
    if not args.no_dedup:
        logger.info("=" * 60)
        logger.info("[2/6] 文本去重 (method=%s) ...", config.data.dedup_method)
        records, dupes = deduplicate_records(records, config.data.dedup_method)
        logger.info("去重前: %d, 去重后: %d, 移除: %d 条",
                     len(records) + dupes, len(records), dupes)

    # ---- 3. Balance ----
    if args.balance:
        logger.info("=" * 60)
        logger.info("[3/6] 类别平衡 (strategy=%s) ...", args.balance_strategy)
        records = balance_records(records, args.balance_strategy, args.seed)

    # ---- 4. Split ----
    split_sum = args.train_ratio + args.dev_ratio
    if split_sum > 1.0:
        logger.warning("train_ratio + dev_ratio = %.2f > 1.0，归一化 ...", split_sum)
        args.train_ratio /= split_sum
        args.dev_ratio /= split_sum

    split_method = "分层(domain × label)" if not args.no_stratified else "纯随机"
    logger.info("=" * 60)
    logger.info("[4/6] 数据集分割 (train=%.0f%%, dev=%.0f%%, test=%.0f%%, method=%s) ...",
                 args.train_ratio * 100, args.dev_ratio * 100,
                 (1 - args.train_ratio - args.dev_ratio) * 100, split_method)

    if args.no_stratified:
        train_set, dev_set, test_set = simple_split(
            records, args.train_ratio, args.dev_ratio, args.seed)
    else:
        train_set, dev_set, test_set = stratified_split(
            records, args.train_ratio, args.dev_ratio, args.seed)

    # ---- 5. Statistics ----
    logger.info("=" * 60)
    logger.info("[5/6] 统计报告 ...")
    stats_list = [
        compute_statistics(train_set, "train"),
        compute_statistics(dev_set, "dev"),
        compute_statistics(test_set, "test"),
    ]
    log_split_statistics(stats_list)

    # ---- 6. Save ----
    logger.info("=" * 60)
    logger.info("[6/6] 保存数据文件 ...")
    save_jsonl(train_set, output_dir / config.paths.train_file_name)
    save_jsonl(dev_set, output_dir / config.paths.dev_file_name)
    save_jsonl(test_set, output_dir / config.paths.test_file_name)

    logger.info("=" * 60)
    logger.info("全部完成! 文件位置: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
