#!/usr/bin/env python3
"""Validate and package the retained experimental run records for Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "addition": 240,
    "addition_dan_kc": 60,
    "addition_diag": 20,
    "addition_disjoint": 160,
    "bandit": 210,
    "bandit_multi": 180,
    "bandit_perm": 720,
    "bandit_punish": 240,
    "comparison": 450,
    "comparison_capacity": 420,
    "comparison_disjoint": 360,
    "gridworld": 440,
    "gridworld_binary": 100,
    "gridworld_layouts": 360,
    "gridworld_layouts_binary": 120,
    "kc_diag": 60,
    "nocalib": 180,
    "plasticity": 505,
    "readouts": 60,
    "readouts_constrained": 40,
    "tictactoe": 340,
    "tictactoe_roles": 50,
}
SHUFFLED = {
    "runs_shuffled_s0": 920,
    "runs_shuffled_s1": 920,
    "runs_shuffled_s2": 920,
}
EXPECTED_TOTAL = 8075


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_and_validate() -> list[Path]:
    files: list[Path] = []
    manifest_total = 0
    errors: list[str] = []

    for group, expected_count in EXPECTED.items():
        folder = ROOT / "runs" / group
        if not folder.is_dir():
            errors.append(f"Missing experiment directory: {folder.relative_to(ROOT)}")
            continue
        manifests = sorted(folder.rglob("manifest.json"))
        if len(manifests) != expected_count:
            errors.append(
                f"{group}: expected {expected_count} manifests, found {len(manifests)}"
            )
        manifest_total += len(manifests)

        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"Cannot read {manifest_path.relative_to(ROOT)}: {exc}")
                continue

            if manifest.get("status") != "ok":
                errors.append(
                    f"{manifest_path.relative_to(ROOT)}: status is {manifest.get('status')!r}, not 'ok'"
                )
            repo_info = manifest.get("repo") or {}
            if repo_info.get("dirty") is not False:
                errors.append(
                    f"{manifest_path.relative_to(ROOT)}: repo.dirty is not false"
                )

            run_dir = manifest_path.parent
            for required in ("manifest.json", "metrics.csv", "summary.json"):
                required_path = run_dir / required
                if not required_path.is_file():
                    errors.append(
                        f"Missing {required} in {run_dir.relative_to(ROOT)}"
                    )

        # Include every file under the experiment directory.
        files.extend(path for path in folder.rglob("*") if path.is_file())

    for folder_name, expected_count in SHUFFLED.items():
        folder = ROOT / folder_name
        if not folder.is_dir():
            errors.append(f"Missing wiring-control directory: {folder_name}")
            continue
        manifests = sorted(folder.rglob("manifest.json"))
        if len(manifests) != expected_count:
            errors.append(
                f"{folder_name}: expected {expected_count} manifests, found {len(manifests)}"
            )
        manifest_total += len(manifests)

        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"Cannot read {manifest_path.relative_to(ROOT)}: {exc}")
                continue
            if manifest.get("status") != "ok":
                errors.append(f"{manifest_path.relative_to(ROOT)}: status is not 'ok'")
            if (manifest.get("repo") or {}).get("dirty") is not False:
                errors.append(f"{manifest_path.relative_to(ROOT)}: repo.dirty is not false")

        files.extend(path for path in folder.rglob("*") if path.is_file())

    splits = ROOT / "runs" / "_splits"
    if not splits.is_dir() or not any(path.is_file() for path in splits.rglob("*")):
        errors.append("runs/_splits is missing or empty")
    else:
        files.extend(path for path in splits.rglob("*") if path.is_file())

    for required in ("DATA_README.md", "DATA_LICENSE.md"):
        path = ROOT / required
        if not path.is_file():
            errors.append(f"Missing required archive file: {required}")
        else:
            files.append(path)

    if manifest_total != EXPECTED_TOTAL:
        errors.append(
            f"Total manifest count should be {EXPECTED_TOTAL}, found {manifest_total}"
        )

    if errors:
        raise SystemExit("DATASET VALIDATION FAILED:\n- " + "\n- ".join(errors))

    unique_files = sorted(set(files))
    print(f"Validated {manifest_total:,} run manifests across {len(EXPECTED) + len(SHUFFLED)} experiment groups.")
    print("All archived manifests have status='ok' and repo.dirty=false.")
    print(f"Collected {len(unique_files):,} files, including saved splits and dataset documentation.")
    return unique_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="dist/fly_mb_dataset_v2.3.zip",
        help="ZIP output path, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the inventory without creating a ZIP archive.",
    )
    args = parser.parse_args()

    files = collect_and_validate()
    if args.check_only:
        print("Check-only mode: no archive created.")
        return

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output, "w:gz") as archive:
    	for path in files:
	        archive.add(path, arcname=path.relative_to(ROOT).as_posix())

    print(f"Created: {output}")
    print(f"Size: {output.stat().st_size:,} bytes")
    print(f"SHA-256: {sha256(output)}")


if __name__ == "__main__":
    main()
