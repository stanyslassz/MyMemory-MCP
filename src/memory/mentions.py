"""Windowed mention_dates management for ACT-R scoring."""

from __future__ import annotations


def add_mention(
    date_iso: str,
    mention_dates: list[str],
    monthly_buckets: dict[str, int],
    window_size: int = 50,
) -> tuple[list[str], dict[str, int]]:
    """Add a mention date and consolidate if window overflows."""
    mention_dates.append(date_iso)
    if len(mention_dates) > window_size:
        mention_dates, monthly_buckets = consolidate_window(
            mention_dates, monthly_buckets, window_size
        )
    return mention_dates, monthly_buckets


def consolidate_window(
    mention_dates: list[str],
    monthly_buckets: dict[str, int],
    window_size: int = 50,
) -> tuple[list[str], dict[str, int]]:
    """Move oldest dates beyond window_size into monthly_buckets."""
    if len(mention_dates) <= window_size:
        return mention_dates, monthly_buckets

    mention_dates.sort()
    overflow = len(mention_dates) - window_size
    to_consolidate = mention_dates[:overflow]
    remaining = mention_dates[overflow:]

    for d in to_consolidate:
        month_key = d[:7]  # "2026-01"
        monthly_buckets[month_key] = monthly_buckets.get(month_key, 0) + 1

    return remaining, monthly_buckets
