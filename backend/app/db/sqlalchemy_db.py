from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine

from ..config import settings

engine = create_engine(
    f"sqlite:///{settings.db_file}",
    connect_args={"check_same_thread": False},
)
metadata = MetaData()

# Column "key" values let PART01 keep its expected attribute names.
problems_table = Table(
    "problems",
    metadata,
    Column("Difficulty", String, key="difficulty"),
    Column("Title", String),
    Column("combindedFrequency", Float, key="combinedFrequency"),
    Column("Category", String),
    Column("Pattern", String),
    Column("pattern_id", Integer),
    Column("subpattern", String),
    Column("subpattern_id", Float),
    Column("url", String),
    Column("curatedListFreq", Integer),
    Column("Notes", String),
    Column("Frequency", Integer),
    Column("tier", Float),
)

pattern_stats_table = Table(
    "pattern_stats",
    metadata,
    Column("pattern", String),
    Column("subpattern", String),
    Column("subpattern_id", Float),
    Column("tier", Float),
    Column("problem_count", Integer),
    Column("total_weight", Float),
)


def execute_stmt(stmt):
    with engine.connect() as conn:
        return conn.execute(stmt).all()


__all__ = ["engine", "metadata", "problems_table", "pattern_stats_table", "execute_stmt"]
