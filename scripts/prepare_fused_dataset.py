#!/usr/bin/env python3
"""Validate ClaimGuard triplets and build leakage-safe FUSED manifests.

Accepted filenames end in ``_1_org``, ``_2_edi`` and ``_3_msk``. The input is
an already-extracted directory. Every validation error stops the run before a
single optimizer step is taken.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


ROLE_RE = re.compile(r"^(?P<group>.+)_(?P<n>[123])_(?P<tag>org|edi|msk)$", re.I)
ROLE_BY_TAG = {"org": "original", "edi": "edited", "msk": "mask"}
ROLE_BY_DIR = {"org": "original", "edi": "edited", "msk": "mask"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Triplet:
    set_id: str
    original: Path
    edited: Path
    mask: Path
    original_sha256: str
    original_dhash: str
    part: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path) -> str:
    with Image.open(path) as image:
        resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixel_source = (
            resized.get_flattened_data() if hasattr(resized, "get_flattened_data")
            else resized.getdata()
        )
        pixels = list(pixel_source)
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return f"{value:016x}"


def validate_image(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"unreadable image: {path} ({exc})") from exc
    if width < 32 or height < 32:
        raise ValueError(f"image is too small ({width}x{height}): {path}")
    return width, height


def infer_part(group: str) -> str:
    prefixes = {
        "f_bmp": "front_bumper", "f_fnd": "front_fender", "light": "headlamp",
        "r_bmp": "rear_bumper", "r_fnd": "rear_fender", "fd": "front_door",
        "bd": "rear_door", "sm": "side_mirror", "bnt": "hood", "trk": "trunk",
    }
    lowered = group.lower()
    return next((label for prefix, label in prefixes.items() if lowered.startswith(prefix)), "other")


def identify_sample(path: Path, root: Path) -> tuple[str, str] | None:
    """Return a stable set id and role for flat or role-separated layouts.

    The supplied dataset uses ``<part>/{org,edi,msk}`` directories. A few of
    its generated filenames also retained an extra ``_1_org`` fragment.  The
    canonical id is derived from the role directory so these harmless naming
    variations do not split one triplet into multiple incomplete samples.
    """
    relative_parent = path.parent.relative_to(root)
    directory_role = ROLE_BY_DIR.get(path.parent.name.lower())
    if directory_role:
        logical_parent = relative_parent.parent.as_posix()
        stem = path.stem
        suffixes = {
            "original": (r"_1_org$",),
            "edited": (r"_2_edi$",),
            "mask": (r"_3_msk$", r"_3msk$"),
        }
        for suffix in suffixes[directory_role]:
            updated = re.sub(suffix, "", stem, flags=re.I)
            if updated != stem:
                stem = updated
                break
        if directory_role in {"edited", "mask"}:
            stem = re.sub(r"_1_org$", "", stem, flags=re.I)
        stem = stem.rstrip("_")
        if not stem:
            return None
        key = f"{logical_parent}/{stem}" if logical_parent != "." else stem
        return key, directory_role

    match = ROLE_RE.match(path.stem)
    if not match:
        return None
    logical_parent = relative_parent.as_posix()
    key = (
        f"{logical_parent}/{match.group('group')}"
        if logical_parent != "." else match.group("group")
    )
    return key, ROLE_BY_TAG[match.group("tag").lower()]


def discover_triplets(root: Path) -> tuple[list[Triplet], list[str]]:
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        identified = identify_sample(path, root)
        if identified is None:
            continue
        key, role = identified
        grouped[key][role].append(path)

    if not grouped:
        raise ValueError(
            "no *_1_org / *_2_edi / *_3_msk files found (subdirectories are supported)"
        )

    errors: list[str] = []
    triplets: list[Triplet] = []
    for set_id, roles in sorted(grouped.items()):
        for role in ("original", "edited", "mask"):
            count = len(roles.get(role, []))
            if count != 1:
                errors.append(f"{set_id}: expected one {role}, found {count}")
        if any(len(roles.get(role, [])) != 1 for role in ("original", "edited", "mask")):
            continue

        original, edited, mask = roles["original"][0], roles["edited"][0], roles["mask"][0]
        try:
            original_size = validate_image(original)
            edited_size = validate_image(edited)
            mask_size = validate_image(mask)
            if original_size != edited_size:
                errors.append(
                    f"WARNING {set_id}: original/edited dimensions differ: "
                    f"{original_size} vs {edited_size}; both are resized independently to 512x512"
                )
            if edited_size != mask_size:
                edited_ratio = edited_size[0] / edited_size[1]
                mask_ratio = mask_size[0] / mask_size[1]
                if abs(edited_ratio - mask_ratio) / edited_ratio > 0.01:
                    raise ValueError(
                        f"edited/mask aspect ratios differ: {edited_size} vs {mask_size}"
                    )
                errors.append(
                    f"WARNING {set_id}: edited/mask dimensions differ: "
                    f"{edited_size} vs {mask_size}; mask is resized with nearest-neighbor sampling"
                )
            original_sha = file_sha256(original)
            if original_sha == file_sha256(edited):
                raise ValueError("original and edited files are byte-identical")
            with Image.open(mask) as image:
                grayscale = image.convert("L")
                lo, hi = grayscale.getextrema()
            if hi <= 127:
                raise ValueError("mask has no manipulated pixels")
            if lo > 127:
                errors.append(f"WARNING {set_id}: mask covers the entire image")
            triplets.append(
                Triplet(
                    set_id=set_id,
                    original=original.resolve(), edited=edited.resolve(), mask=mask.resolve(),
                    original_sha256=original_sha, original_dhash=dhash(original),
                    part=infer_part(Path(set_id).name),
                )
            )
        except ValueError as exc:
            errors.append(f"{set_id}: {exc}")

    hard_errors = [item for item in errors if not item.startswith("WARNING ")]
    if hard_errors:
        preview = "\n".join(f"  - {item}" for item in hard_errors[:30])
        suffix = f"\n  ... and {len(hard_errors) - 30} more" if len(hard_errors) > 30 else ""
        raise ValueError(f"dataset validation failed:\n{preview}{suffix}")
    return triplets, [item for item in errors if item.startswith("WARNING ")]


def assign_splits(triplets: list[Triplet], seed: int) -> dict[str, list[Triplet]]:
    # Exact and perceptual duplicates of an original stay together. This blocks
    # near-identical source photos from leaking into validation/test.
    leakage_groups: dict[str, list[Triplet]] = defaultdict(list)
    for triplet in triplets:
        leakage_groups[triplet.original_dhash].append(triplet)
    keys = sorted(leakage_groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 3:
        raise ValueError("at least three distinct original-image groups are required")

    train_end = max(1, int(len(keys) * 0.80))
    val_end = max(train_end + 1, int(len(keys) * 0.90))
    val_end = min(val_end, len(keys) - 1)
    selected = {
        "train": keys[:train_end],
        "val": keys[train_end:val_end],
        "test": keys[val_end:],
    }
    return {
        name: [item for key in group_keys for item in leakage_groups[key]]
        for name, group_keys in selected.items()
    }


def manifest_rows(triplets: list[Triplet]) -> list[list[str | None]]:
    rows: list[list[str | None]] = []
    for item in sorted(triplets, key=lambda value: value.set_id):
        rows.append([str(item.original), None, "claimguard_real"])
        rows.append([str(item.edited), str(item.mask), "claimguard_partial_fake"])
    return rows


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="already-extracted triplet directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-sets", type=int, default=501,
        help="required complete triplet count; use 0 only for an intentional subset",
    )
    args = parser.parse_args()

    data = args.data.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not data.is_dir():
        raise FileNotFoundError(f"training image directory does not exist: {data}")
    source_root = data

    triplets, warnings = discover_triplets(source_root)
    if args.expected_sets and len(triplets) != args.expected_sets:
        raise ValueError(
            f"expected {args.expected_sets} complete triplets from the specification, "
            f"but found {len(triplets)}"
        )
    splits = assign_splits(triplets, args.seed)
    for name, members in splits.items():
        write_json_atomic(output / f"{name}.json", manifest_rows(members))

    split_for_set = {item.set_id: name for name, members in splits.items() for item in members}
    report = {
        "source": str(source_root.resolve()),
        "seed": args.seed,
        "expected_sets": args.expected_sets,
        "sets": len(triplets),
        "images": len(triplets) * 2,
        "splits": {
            name: {"sets": len(members), "images": len(members) * 2}
            for name, members in splits.items()
        },
        "parts": dict(sorted(Counter(item.part for item in triplets).items())),
        "warnings": warnings,
        "assignments": split_for_set,
    }
    write_json_atomic(output / "dataset_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
