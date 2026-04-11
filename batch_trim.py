"""
Batch-trim all SLEEP-Cassette PSG EDF files using their hypnograms.

Usage:
    python batch_trim.py --src "D:/path/to/sleep-cassette" --dst "D:/path/to/trimmed"

Options:
    --pad-min   Extra minutes to keep before/after sleep window (default: 5)
    --workers   Parallel workers (default: 1 — safe for RAM)
    --dry-run   Print what would be done without writing any files
"""

import argparse
import glob
import os
import sys
import traceback
from pathlib import Path

import mne


SLEEP_STAGE_LABELS = {
    "Sleep stage 1", "Sleep stage 2", "Sleep stage 3",
    "Sleep stage 4", "Sleep stage R",
}
EEG_CHANNEL = "EEG Fpz-Cz"


# ---------------------------------------------------------------------------
# Core trim logic (same as trim_edf.py)
# ---------------------------------------------------------------------------

def find_sleep_window(hypno_path: str):
    """Return (t_start_sec, t_end_sec) of the sleep window from a hypnogram."""
    annot = mne.read_annotations(hypno_path)
    sleep_onsets = [
        annot.onset[i]
        for i, desc in enumerate(annot.description)
        if desc in SLEEP_STAGE_LABELS
    ]
    if not sleep_onsets:
        raise ValueError(f"No sleep stages found in {hypno_path}. "
                         f"Labels: {sorted(set(annot.description))}")
    return min(sleep_onsets), max(sleep_onsets) + 30  # +30 s for last epoch


def trim_one(psg_path: str, hypno_path: str, out_path: str, pad_min: float = 5.0):
    """Trim one PSG file to its sleep window and save."""
    first_sleep, last_sleep = find_sleep_window(hypno_path)
    t_start = max(0.0, first_sleep - pad_min * 60)
    t_end   = last_sleep + pad_min * 60

    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    t_end = min(t_end, raw.times[-1])
    raw.crop(tmin=t_start, tmax=t_end)

    # Keep only EEG Fpz-Cz — other channels have extreme ranges
    # that overflow EDF's 8-char header field.
    if EEG_CHANNEL in raw.ch_names:
        raw.pick_channels([EEG_CHANNEL])
    else:
        eeg = raw.copy().pick_types(eeg=True)
        if eeg.ch_names:
            raw.pick_channels([eeg.ch_names[0]])

    mne.export.export_raw(out_path, raw, fmt="edf",
                          physical_range="channelwise", overwrite=True)
    kept_min = (t_end - t_start) / 60
    return kept_min


# ---------------------------------------------------------------------------
# Pairing logic
# ---------------------------------------------------------------------------

def find_pairs(src_dir: str):
    """
    Match every *-PSG.edf with its *-Hypnogram.edf.
    The hypnogram ID letter varies (SC4001EC, SC4001EH, …) so we glob.
    """
    psg_files = sorted(glob.glob(os.path.join(src_dir, "*-PSG.edf")))
    pairs = []
    missing = []

    for psg in psg_files:
        # e.g. SC4001E0-PSG.edf → prefix SC4001E
        base    = Path(psg).stem          # SC4001E0-PSG
        subject = base[:7]                # SC4001E  (first 7 chars)
        pattern = os.path.join(src_dir, f"{subject}*-Hypnogram.edf")
        matches = glob.glob(pattern)

        if matches:
            pairs.append((psg, matches[0]))
        else:
            missing.append(psg)

    return pairs, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch-trim SLEEP-Cassette EDF files.")
    p.add_argument("--src",      required=True, help="Source folder with PSG + Hypnogram EDFs")
    p.add_argument("--dst",      required=True, help="Output folder for trimmed EDFs")
    p.add_argument("--pad-min",  type=float, default=5.0,
                   help="Padding before/after sleep window in minutes (default: 5)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print plan without writing files")
    return p.parse_args()


def main():
    args = parse_args()

    src = args.src
    dst = args.dst

    if not os.path.isdir(src):
        print(f"ERROR: source folder not found: {src}")
        sys.exit(1)

    os.makedirs(dst, exist_ok=True)

    pairs, missing = find_pairs(src)
    print(f"Found {len(pairs)} PSG+Hypnogram pairs  ({len(missing)} PSG files without a hypnogram)")

    if missing:
        print("  Skipping (no hypnogram):")
        for f in missing:
            print(f"    {os.path.basename(f)}")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for psg, hypno in pairs:
            out = os.path.join(dst, os.path.basename(psg).replace("-PSG.edf", "-trimmed.edf"))
            print(f"  {os.path.basename(psg)}  +  {os.path.basename(hypno)}  →  {os.path.basename(out)}")
        return

    print(f"\nTrimming {len(pairs)} files → {dst}\n")

    ok, failed = 0, []

    for i, (psg, hypno) in enumerate(pairs, 1):
        out_name = os.path.basename(psg).replace("-PSG.edf", "-trimmed.edf")
        out_path = os.path.join(dst, out_name)

        if os.path.exists(out_path):
            print(f"[{i:3}/{len(pairs)}] SKIP (exists)  {out_name}")
            ok += 1
            continue

        try:
            kept = trim_one(psg, hypno, out_path, pad_min=args.pad_min)
            print(f"[{i:3}/{len(pairs)}] OK  {out_name}  ({kept:.0f} min)")
            ok += 1
        except Exception as e:
            print(f"[{i:3}/{len(pairs)}] FAIL {os.path.basename(psg)}: {e}")
            failed.append((os.path.basename(psg), str(e)))

    print(f"\nDone: {ok} succeeded, {len(failed)} failed")
    if failed:
        print("Failed files:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
