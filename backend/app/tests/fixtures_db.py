from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Tuple

ProblemRow = Tuple[
    str,   # Difficulty
    str,   # Title
    float, # combindedFrequency
    str,   # Category
    str,   # Pattern
    int,   # pattern_id
    str,   # subpattern
    float, # subpattern_id
    str,   # url
    int,   # curatedListFreq
    str,   # Notes
    int,   # Frequency
    float, # tier
]


def seed_problems() -> List[ProblemRow]:
    return [
        ("easy", "TP Array A", 10.0, "arrays", "two pointers", 10, "converging", 10.1, "http://a", 1, "", 1, 1.0),
        ("easy", "TP Array B", 9.0, "arrays", "two pointers", 10, "parallel", 10.2, "http://b", 1, "", 1, 1.0),
        ("medium", "BST Intro", 8.0, "trees", "tree - binary search tree", 44, "n/a", 44.1, "http://c", 1, "", 1, 3.0),
        ("hard", "DFS Deep", 7.0, "trees", "tree - depth first search", 45, "n/a", 45.1, "http://d", 1, "", 1, 3.0),
        ("medium", "Prefix Sum", 6.0, "arrays", "prefix array", 8, "standard", 8.2, "http://e", 1, "", 1, 1.5),
        ("easy", "Intervals 1", 5.0, "intervals", "intervals", 5, "conflicts", 5.1, "http://f", 1, "", 1, 2.5),
    ]

PatternStatsRow = Tuple[str, str, float, float, int, float]


def seed_pattern_stats() -> List[PatternStatsRow]:
    return [
        ("two pointers", "converging", 10.1, 1.0, 20, 4.99),
        ("two pointers", "parallel", 10.2, 1.0, 15, 3.50),
        ("tree - binary search tree", "n/a", 44.1, 3.0, 10, 2.00),
        ("tree - depth first search", "n/a", 45.1, 3.0, 12, 2.40),
        ("prefix array", "standard", 8.2, 1.5, 2, 0.50),
        ("intervals", "conflicts", 5.1, 2.5, 4, 1.00),
    ]


def _create_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE problems (
            Difficulty TEXT,
            Title TEXT,
            combindedFrequency REAL,
            Category TEXT,
            Pattern TEXT,
            pattern_id INTEGER,
            subpattern TEXT,
            subpattern_id REAL,
            url TEXT,
            curatedListFreq INT,
            Notes TEXT,
            Frequency INT,
            tier REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE pattern_stats (
            pattern TEXT,
            subpattern TEXT,
            subpattern_id REAL,
            tier REAL,
            problem_count INT,
            total_weight REAL
        )
        """
    )
    conn.commit()

def build_test_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test_engine.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        _create_tables(conn)
        conn.executemany(
            """
            INSERT INTO problems (
                Difficulty, Title, combindedFrequency, Category, Pattern, pattern_id, subpattern,
                subpattern_id, url, curatedListFreq, Notes, Frequency, tier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            seed_problems(),
        )
        conn.executemany(
            """
            INSERT INTO pattern_stats (
                pattern, subpattern, subpattern_id, tier, problem_count, total_weight
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            seed_pattern_stats(),
        )
        conn.commit()
    finally:
        conn.close()

    return db_path


__all__ = ["build_test_db", "seed_problems", "seed_pattern_stats"]
