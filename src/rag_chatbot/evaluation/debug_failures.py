from pathlib import Path

import pandas as pd
import structlog


log = structlog.get_logger(__name__)


# Each tuple: (metric_column, score_threshold, fix_advice)
CHECKS = [
    (
        "faithfulness",
        0.70,
        "Strengthen system prompt -- LLM is using outside knowledge",
    ),
    (
        "context_precision",
        0.60,
        "Reduce CHUNK_SIZE -- chunks encode too many topics at once",
    ),
    (
        "context_recall",
        0.60,
        "Increase CHUNK_OVERLAP -- answer spans a chunk boundary",
    ),
    (
        "answer_relevancy",
        0.65,
        "Increase TOP_K -- not enough context is reaching the LLM",
    ),
]


def debug_failures(
    csv_path: str = "evaluation/ragas_results.csv",
) -> None:
    """
    Load the RAGAS results CSV and print low-scoring samples per metric.

    Args:
        csv_path: Path to the CSV written by evaluate.py.
    """
    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        print("Run evaluate.py first to generate results.")
        return

    df = pd.read_csv(csv_path)

    print(f"\nTotal evaluated samples: {len(df)}\n")

    for col, threshold, advice in CHECKS:
        if col not in df.columns:
            continue

        low = df[df[col] < threshold].sort_values(col)

        print("-" * 60)
        print(
            f"LOW {col.upper()} "
            f"(below {threshold}): {len(low)} samples"
        )
        print(f"FIX: {advice}")

        for _, row in low.head(2).iterrows():
            print(f"\n Question : {row.get('question', 'N/A')}")
            print(
                f" Answer   : "
                f"{str(row.get('answer', ''))[:120]}..."
            )
            print(f" Score    : {row[col]:.3f}")

        print()


if __name__ == "__main__":
    debug_failures()
