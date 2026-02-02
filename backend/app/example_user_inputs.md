Example user inputs (from `example_user_inputs.py`), translated to JSON shape:

```json
{
  "user_requested_roadmap_length": 120,
  "user_difficulty_selection": [
    ["easy", 1.0],
    ["medium", 1.0],
    ["hard", 0.7]
  ],
  "user_priority_patterns": [
    ["Prefix Array", 8],
    ["Intervals", 5]
  ],
  "user_excluded_patterns": [
    ["Marking and Index Tricks", 6],
    ["Grid Explorer", 13],
    ["Geometry", 36]
  ]
}
```

Notes:
- JSON has no tuples, so these are lists-of-lists in the request payload.
- If we add `chunk_size` or `problems_per_day`, it would live at the top level.
