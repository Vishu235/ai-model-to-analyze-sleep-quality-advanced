"""
Crop an EDF recording to the sleep window and save a new EDF.

Mode 1 — Auto-detect from hypnogram (most accurate):
    python trim_edf.py input.edf output.edf --hypnogram SC4001EC-Hypnogram.edf

Mode 2 — Inspect only (find out what's in the file before cropping):
    python trim_edf.py input.edf --inspect

Mode 3 — Manual crop by offset:
    python trim_edf.py input.edf output.edf --offset-min 60 --duration-min 480
"""

import argparse
import sys
import mne


SLEEP_STAGE_LABELS = {"Sleep stage 1", "Sleep stage 2", "Sleep stage 3",
                      "Sleep stage 4", "Sleep stage R"}


def inspect(edf_path: str):
    """Print a summary of what's inside the EDF."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    total_min = raw.times[-1] / 60
    print(f"\n--- EDF Inspection: {edf_path} ---")
    print(f"  Duration   : {total_min:.1f} min  ({total_min/60:.2f} hrs)")
    print(f"  Sample rate: {raw.info['sfreq']} Hz")
    print(f"  Channels   : {raw.ch_names}")
    if raw.annotations and len(raw.annotations) > 0:
        unique = sorted(set(raw.annotations.description))
        print(f"  Annotations: {len(raw.annotations)} total — {unique}")
    else:
        print("  Annotations: none")
    print()


def auto_crop_from_hypnogram(edf_path: str, hypno_path: str, output_path: str,
                              pad_min: float = 5.0):
    """Detect sleep onset and last sleep epoch from hypnogram, then crop."""
    print(f"Reading hypnogram: {hypno_path}")
    annot = mne.read_annotations(hypno_path)

    sleep_onsets = []
    for i, desc in enumerate(annot.description):
        if desc in SLEEP_STAGE_LABELS:
            sleep_onsets.append(annot.onset[i])

    if not sleep_onsets:
        print("ERROR: No sleep stages found in hypnogram.")
        print(f"  Labels present: {sorted(set(annot.description))}")
        sys.exit(1)

    first_sleep = min(sleep_onsets)
    last_sleep  = max(sleep_onsets) + 30  # each epoch is 30 s

    t_start = max(0.0, first_sleep - pad_min * 60)
    t_end   = last_sleep + pad_min * 60

    print(f"  First sleep epoch : {first_sleep/60:.1f} min into recording")
    print(f"  Last  sleep epoch : {last_sleep/60:.1f} min into recording")
    print(f"  Crop window       : {t_start/60:.1f} → {t_end/60:.1f} min  "
          f"(+/- {pad_min:.0f} min padding)")

    _crop_and_save(edf_path, output_path, t_start, t_end)


def manual_crop(edf_path: str, output_path: str,
                offset_min: float, duration_min: float):
    """Crop by a fixed offset + duration."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    t_start = offset_min * 60
    t_end   = min(t_start + duration_min * 60, raw.times[-1])
    print(f"Manual crop: {t_start/60:.1f} → {t_end/60:.1f} min  "
          f"({(t_end-t_start)/60:.1f} min kept)")
    _crop_and_save(edf_path, output_path, t_start, t_end)


EEG_CHANNEL = "EEG Fpz-Cz"


def _crop_and_save(edf_path: str, output_path: str, t_start: float, t_end: float):
    print(f"Loading {edf_path} ...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.crop(tmin=t_start, tmax=t_end)

    # Keep only the EEG channel — other channels (temp, airflow, etc.)
    # have extreme physical ranges that exceed EDF's 8-char header limit.
    if EEG_CHANNEL in raw.ch_names:
        raw.pick_channels([EEG_CHANNEL])
        print(f"  Keeping channel: {EEG_CHANNEL}")
    else:
        # Fall back to first EEG-type channel
        raw_eeg = raw.copy().pick_types(eeg=True)
        if raw_eeg.ch_names:
            raw.pick_channels([raw_eeg.ch_names[0]])
            print(f"  '{EEG_CHANNEL}' not found — using: {raw_eeg.ch_names[0]}")
        else:
            print(f"  WARNING: no EEG channel found, saving all channels")

    mne.export.export_raw(output_path, raw, fmt="edf", physical_range="channelwise", overwrite=True)
    kept_min = (t_end - t_start) / 60
    print(f"Saved {kept_min:.1f} min → {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Crop an EDF to the sleep window.")
    p.add_argument("input",          help="Source EDF file")
    p.add_argument("output",         nargs="?", help="Output EDF file (omit for --inspect)")

    p.add_argument("--inspect",      action="store_true",
                   help="Just print file info, don't crop")
    p.add_argument("--hypnogram",    help="Hypnogram EDF for auto sleep-window detection")
    p.add_argument("--offset-min",   type=float, default=0,
                   help="Minutes from start to begin crop (default: 0)")
    p.add_argument("--duration-min", type=float, default=480,
                   help="Minutes to keep (default: 480 = 8 hrs)")
    p.add_argument("--pad-min",      type=float, default=5,
                   help="Extra minutes to keep before/after sleep (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.inspect:
        inspect(args.input)
        return

    if args.output is None:
        print("ERROR: provide an output path, or use --inspect")
        sys.exit(1)

    if args.hypnogram:
        auto_crop_from_hypnogram(args.input, args.hypnogram, args.output,
                                 pad_min=args.pad_min)
    else:
        manual_crop(args.input, args.output, args.offset_min, args.duration_min)


if __name__ == "__main__":
    main()
