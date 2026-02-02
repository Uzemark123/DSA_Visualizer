#!/usr/bin/env python
"""
Convert an Obsidian Excalidraw .md file (compressed-json block) into .excalidraw JSON.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path

KEY_STR = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
BASE_REVERSE = {c: i for i, c in enumerate(KEY_STR)}


def _decompress(length, reset_value, get_next_value):
    dictionary = [0, 1, 2]
    enlarge_in = 4
    dict_size = 4
    num_bits = 3

    data_val = get_next_value(0)
    data_position = reset_value
    data_index = 1

    def read_bits(n_bits):
        nonlocal data_val, data_position, data_index
        bits = 0
        power = 1
        max_power = 1 << n_bits
        while power != max_power:
            resb = data_val & data_position
            data_position >>= 1
            if data_position == 0:
                data_position = reset_value
                if data_index < length:
                    data_val = get_next_value(data_index)
                data_index += 1
            if resb > 0:
                bits |= power
            power <<= 1
        return bits

    next_bits = read_bits(2)
    if next_bits == 0:
        c = chr(read_bits(8))
    elif next_bits == 1:
        c = chr(read_bits(16))
    elif next_bits == 2:
        return ""
    else:
        return None

    dictionary.append(c)
    w = c
    result = [c]

    while True:
        if data_index > length:
            return ""
        c = read_bits(num_bits)
        if c == 0:
            dictionary.append(chr(read_bits(8)))
            c = dict_size
            dict_size += 1
            enlarge_in -= 1
        elif c == 1:
            dictionary.append(chr(read_bits(16)))
            c = dict_size
            dict_size += 1
            enlarge_in -= 1
        elif c == 2:
            return "".join(result)

        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1

        if c < len(dictionary) and dictionary[c] is not None:
            entry = dictionary[c]
        elif c == dict_size:
            entry = w + w[0]
        else:
            return None

        result.append(entry)
        dictionary.append(w + entry[0])
        dict_size += 1
        enlarge_in -= 1
        w = entry

        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1


def decompress_from_base64(input_str):
    if not input_str:
        return ""

    def get_next_value(index):
        return BASE_REVERSE.get(input_str[index], 0)

    return _decompress(len(input_str), 32, get_next_value)


def extract_compressed_block(markdown_text):
    match = re.search(r"```compressed-json\s*(.*?)\s*```", markdown_text, re.S)
    if not match:
        raise ValueError("compressed-json block not found")
    return "".join(match.group(1).splitlines()).strip()


def extract_embedded_files(markdown_text):
    match = re.search(r"## Embedded Files\s*(.*?)(?:\n%%|\n## Drawing)", markdown_text, re.S)
    if not match:
        return []
    block = match.group(1)
    entries = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        file_match = re.match(r"^([0-9a-f]{40}):\s*\[\[(.+?)\]\]$", line, re.I)
        if not file_match:
            continue
        entries.append((file_match.group(1), file_match.group(2)))
    return entries


def build_files_map(md_dir, embedded_entries):
    if not embedded_entries:
        return {}
    files_map = {}
    now = int(time.time() * 1000)
    for file_id, filename in embedded_entries:
        candidates = [md_dir / filename, md_dir / "images" / filename]
        path = next((p for p in candidates if p.exists()), None)
        if not path:
            raise SystemExit(f"Embedded file not found: {filename}")
        data = path.read_bytes()
        suffix = path.suffix.lower()
        if suffix == ".svg":
            mime_type = "image/svg+xml"
        elif suffix == ".png":
            mime_type = "image/png"
        else:
            raise SystemExit(f"Unsupported embedded file type: {path.name}")
        encoded = base64.b64encode(data).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        files_map[file_id] = {
            "id": file_id,
            "dataURL": data_url,
            "mimeType": mime_type,
            "created": now,
            "lastRetrieved": now,
        }
    return files_map


def main():
    parser = argparse.ArgumentParser(description="Convert Obsidian Excalidraw .md to .excalidraw JSON")
    parser.add_argument("input", help="Path to .md file")
    parser.add_argument(
        "--out",
        help="Output .excalidraw path (defaults to same name as input)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    markdown = in_path.read_text(encoding="utf-8")
    compressed = extract_compressed_block(markdown)
    json_text = decompress_from_base64(compressed)
    if not json_text:
        raise SystemExit("Decompression failed or produced empty output")

    # Normalize surrogate pairs (emoji) into valid Unicode.
    json_text = json_text.encode("utf-16", "surrogatepass").decode("utf-16")

    doc = json.loads(json_text)
    embedded_entries = extract_embedded_files(markdown)
    embedded_files = build_files_map(in_path.parent, embedded_entries)
    if embedded_files:
        existing_files = doc.get("files", {}) or {}
        doc["files"] = {**existing_files, **embedded_files}

    out_path = Path(args.out) if args.out else in_path.with_suffix(".excalidraw")
    out_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
