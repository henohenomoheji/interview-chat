# -*- coding: utf-8 -*-
"""
cafe_pl_data データセット作成スクリプト
- 期間: 2024/01 〜 2025/12
- 値はすべて 0 以上
- 出力: cafe_pl_data.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd


# =========================
# マスタ定義
# =========================

CATEGORIES = ["ラテ", "コーヒー", "カプチーノ", "紅茶", "抹茶"]

CHANNELS = ["コンビニ", "直営店", "EC", "卸"]

FIN_ITEMS = [
    "Royalty Revenue",
    "Cash Discounts",
    "Price Discounts",
    "Bid Allowance",
    "Spoilage",
    "Coupon Redemption",
    "Promotion",
    "Slottting",
]

VERSIONS = ["当年", "前年", "予定"]

UNITS = ["単価", "合計金額", "パーセンテージ"]


@dataclass(frozen=True)
class ValueRanges:
    unit_price_min: float = 80.0
    unit_price_max: float = 260.0
    total_amount_min: float = 5_000.0
    total_amount_max: float = 2_000_000.0
    percent_min: float = 0.0
    percent_max: float = 30.0  # %


def generate_value(
    rng: np.random.Generator,
    unit: str,
    version: str,
    ranges: ValueRanges,
) -> float:
    """
    値生成（すべて 0 以上）
    """

    # 単位別レンジ
    if unit == "単価":
        base = rng.uniform(ranges.unit_price_min, ranges.unit_price_max)
    elif unit == "合計金額":
        base = rng.uniform(ranges.total_amount_min, ranges.total_amount_max)
    elif unit == "パーセンテージ":
        base = rng.uniform(ranges.percent_min, ranges.percent_max)
    else:
        raise ValueError(unit)

    # バージョン差分
    version_factor = {
        "前年": 0.95,
        "当年": 1.00,
        "予定": 1.05,
    }[version]

    value = base * version_factor

    # 丸め
    return round(float(value), 2)


def main() -> None:
    rng = np.random.default_rng(42)
    ranges = ValueRanges()

    months = pd.date_range("2024-01-01", "2025-12-01", freq="MS")

    rows = []
    for dt in months:
        for category, channel, fin_item, version, unit in product(
            CATEGORIES, CHANNELS, FIN_ITEMS, VERSIONS, UNITS
        ):
            rows.append(
                {
                    "年": int(dt.year),
                    "月": int(dt.month),
                    "カテゴリ": category,
                    "チャネル": channel,
                    "財務項目": fin_item,
                    "バージョン": version,
                    "単位": unit,
                    "値": generate_value(rng, unit, version, ranges),
                }
            )

    df = pd.DataFrame(
        rows,
        columns=["年", "月", "カテゴリ", "チャネル", "財務項目", "バージョン", "単位", "値"],
    )

    df.to_csv("cafe_pl_data.csv", index=False, encoding="utf-8-sig")

    print(f"✅ 出力完了: {len(df):,} 行")
    print(df.head(10))


if __name__ == "__main__":
    main()
