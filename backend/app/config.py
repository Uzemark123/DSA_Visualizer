import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    port: int = int(os.getenv("PORT", "3001"))
    env: str = os.getenv("NODE_ENV", "development")
    db_file: str = os.getenv(
        "DB_FILE",
        os.path.join(os.path.dirname(__file__), "db", "production_database.sqlite"),
    )


settings = Settings()


# Roadmap domain constants (moved from engine/config.py).
ALL_PROBLEMS_AND_SUBPATTERN_IDS_IN_DB = [
    ("Prefix Array", 8),
    ("Combinatorial Generation", 11),
    ("Tree- Breath first search", 47),
    ("Marking and Index Tricks", 6),
    ("Constructive Backtracking", 12),
    ("Sort And Sweep ", 24),
    ("MonoStack", 7),
    ("Intervals", 5),
    ("DAGs", 18),
    ("Tree - Binary Search tree", 44),
    ("Regions and Connectivity", 19),
    ("Geometry", 36),
    ("bittricks", 2),
    ("Linear DP", 16),
    ("String DP", 17),
    ("Tree - Depth first seach", 45),
    ("Shortest Path", 20),
    ("Tree Construction", 46),
    ("Static Hashing", 41),
    ("Trie-Based", 42),
    ("Stack Based", 40),
    ("Knapsack family", 15),
    ("Huffman-Style (PQ)", 22),
    ("Design", 50),
    ("Two Pointers", 10),
    ("Sliding Window", 9),
    ("Circular Arrays", 3),
    ("Circular Strings", 51),
    ("Grid DP", 14),
    ("Heaps", 4),
    ("One-Pass Greedy", 23),
    ("Classic", 25),
    ("Binary Search on Answer Space", 1),
    ("Two Pointers", 43),
    ("Grid Explorer", 13),
    ("miscellaneous", 34),
    ("Structural Validations", 21),
    ("Sliding w + freq map", 39),
]

TIERs_AS_ARRAY = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]

TIERS_TO_PATTERN = {
    1: {43.1, 43.2, 10.1, 41.2, 10.2, 41.1, 41.3},
    1.5: {2.1, 39.1, 8.1, 8.2, 9.2, 9.3, 6.1, 9.1},
    2: {34.1, 40.1, 40.3, 40.2, 25.1, 4.1, 51.1},
    2.5: {3.1, 3.3, 23.1, 5.1, 3.2, 5.2},
    3: {7.1, 22.1, 44.1, 45.1, 36.1, 47.1, 24.1},
    3.5: {42.1, 50.1, 46.1, 19.2},
    4: {16.2, 20.1, 16.1, 19.1, 18.1},
    4.5: {14.1, 11.4, 11.3, 11.1, 15.2, 13.1, 15.1, 11.2, 15.3, 21.1},
    5: {18.2, 12.1, 21.3, 21.2},
    5.5: {1.1, 17.4, 17.3, 17.1, 17.2, 20.2},
}

PATTERN_TO_TIER = {
    43.1: 1,
    43.2: 1,
    10.1: 1,
    41.2: 1,
    10.2: 1,
    41.1: 1,
    41.3: 1,
    2.1: 1.5,
    39.1: 1.5,
    8.1: 1.5,
    8.2: 1.5,
    9.2: 1.5,
    9.3: 1.5,
    6.1: 1.5,
    9.1: 1.5,
    34.1: 2,
    40.1: 2,
    40.3: 2,
    40.2: 2,
    25.1: 2,
    4.1: 2,
    3.1: 2.5,
    3.3: 2.5,
    23.1: 2.5,
    5.1: 2.5,
    3.2: 2.5,
    5.2: 2.5,
    51.1: 2,
    7.1: 3,
    22.1: 3,
    44.1: 3,
    45.1: 3,
    36.1: 3,
    47.1: 3,
    24.1: 3,
    42.1: 3.5,
    50.1: 3.5,
    46.1: 3.5,
    19.2: 3.5,
    16.2: 4,
    20.1: 4,
    16.1: 4,
    19.1: 4,
    18.1: 4,
    14.1: 4.5,
    11.4: 4.5,
    11.3: 4.5,
    11.1: 4.5,
    15.2: 4.5,
    13.1: 4.5,
    15.1: 4.5,
    11.2: 4.5,
    15.3: 4.5,
    21.1: 4.5,
    18.2: 5,
    12.1: 5,
    21.3: 5,
    21.2: 5,
    1.1: 5.5,
    17.4: 5.5,
    17.3: 5.5,
    17.1: 5.5,
    17.2: 5.5,
    20.2: 5.5,
}

PATTERN_ID_TO_SUBPATTERN_ID = {
    1: [1.1],  # binary search on answer space: n/a
    2: [2.1],  # bittricks: n/a
    3: [3.1, 3.2, 3.3],  # circular arrays: rotated arrays, gas station, rotate array
    51: [51.1],  # circular strings: n/a
    4: [4.1],  # heaps: n/a
    5: [5.1, 5.2],  # intervals: conflicts, merge/insert/intersections
    6: [6.1],  # marking and index tricks: n/a
    7: [7.1],  # monostack: n/a
    8: [8.1, 8.2],  # prefix array: curr - k, standard
    9: [9.1, 9.2, 9.3],  # sliding window: counting subarrays, fixed-size, standard validity
    10: [10.1, 10.2],  # two pointers: converging, parallel
    11: [11.1, 11.2, 11.3, 11.4],  # combinatorial generation: combination sum, combinations, permutations, subsets
    12: [12.1],  # constructive backtracking: n/a
    13: [13.1],  # grid explorer: n/a
    14: [14.1],  # grid dp: n/a
    15: [15.1, 15.2, 15.3],  # knapsack family: feasibility, unbounded, unbounded
    16: [16.1, 16.2],  # linear dp: linear min-max, linear ways
    17: [17.1, 17.2, 17.3, 17.4],  # string dp: edit distance, palindromic, pattern matching, subsequence
    18: [18.1, 18.2],  # dags: cycle detection, topological ordering
    19: [19.1, 19.2],  # regions and connectivity: connected components, flood fill/islands
    20: [20.1, 20.2],  # shortest path: unweighted (bfs), weighted (dijkstra)
    21: [21.1, 21.2, 21.3],  # structural validations: bipartite, constraint propagation, cycle check (undirected)
    22: [22.1],  # huffman-style (pq): n/a
    23: [23.1],  # one-pass greedy: n/a
    24: [24.1],  # sort and sweep: n/a
    25: [25.1],  # classic: n/a
    34: [34.1],  # miscellaneous: n/a
    36: [36.1],  # geometry: n/a
    39: [39.1],  # sliding w + freq map: n/a
    40: [40.1, 40.2, 40.3],  # stack based: bracket checking, history stack, nested builder
    41: [41.1, 41.2, 41.3],  # static hashing: frequency, membership, signature hashing
    42: [42.1],  # trie-based: n/a
    43: [43.1, 43.2],  # two pointers: converging, parallel
    44: [44.1],  # tree - binary search tree: n/a
    45: [45.1],  # tree - depth first search: n/a
    46: [46.1],  # tree construction: n/a
    47: [47.1],  # tree - breath first search: n/a
    50: [50.1],  # design: n/a
}


STRUCTURED_RATIO = 0.8
DEFAULT_PRIORITY_MULTIPLIER = 1.5
