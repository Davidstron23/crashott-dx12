r"""Compares the screenshots of two run folders (tools/run_game.py) pixel by pixel.

    python tools\perf\compare_runs.py build\vulkan\baseline\<run> build\vulkan\vulkan\<run> [--diff out_dir]

Screenshots are paired in order (their frame numbers can differ between runs by a frame or two).
For each pair: mean absolute difference per channel (0-255), the share of pixels that differ by
more than 8, and the largest difference. --diff writes amplified difference images.

Tolerance mode (spec 002, P3) judges a candidate run against two runs of the baseline:

    python tools\perf\compare_runs.py --tolerance <base_1> <base_2> <candidate> [--floor 0.1] [--diff out_dir]

For each anchored capture i, noise_i = "% of pixels > 8" between base_1 and base_2; the candidate passes
when every capture differs from base_1 by at most max(2 * noise_i, floor) % of pixels (its own threshold,
so one animated capture does not loosen the others), with the same number of captures at the same sizes.
Only the captures anchored to events (SHOT) are judged; the ones taken at frame numbers
(`screenshots = [N]` in the profile, read from the run's settings.json) go to an "eyeball" list.
Exit code 0 = pass, 1 = fail.
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError:
    sys.exit("needs numpy and Pillow (python -m pip install -r tools\\requirements.txt)")


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def changed_share(image_a, image_b):
    """% of pixels where some channel differs by more than 8."""
    return (np.abs(image_a - image_b).max(axis=2) > 8).mean() * 100.0


def split_captures(folder):
    """(anchored SHOT captures, frame-number captures) of a run folder, each sorted."""
    folder = Path(folder)
    frames = set()
    try:
        frames = {int(n) for n in json.loads((folder / "settings.json").read_text(encoding="utf-8"))["settings"].get("screenshots") or []}
    except (OSError, ValueError, KeyError):
        pass  # no settings.json: every capture counts as anchored
    anchored, eyeball = [], []
    for shot in sorted(folder.glob("screenshot_*.png")):
        match = re.search(r"(\d+)", shot.stem)
        (eyeball if match and int(match.group(1)) in frames else anchored).append(shot)
    return anchored, eyeball


def tolerance(base_1, base_2, candidate, floor, diff_folder):
    runs = [split_captures(folder) for folder in (base_1, base_2, candidate)]
    (anchored_1, _), (anchored_2, _), (anchored_c, eyeball_c) = runs
    if not anchored_1 or not anchored_2 or not anchored_c:
        print("FAIL: no anchored (SHOT) captures in one of the folders")
        return 1
    if not len(anchored_1) == len(anchored_2) == len(anchored_c):
        print(f"FAIL: different number of captures: {len(anchored_1)} / {len(anchored_2)} / {len(anchored_c)}")
        return 1
    if diff_folder:
        Path(diff_folder).mkdir(parents=True, exist_ok=True)
    ok = True
    noises, shares = [], []
    for a, b, c in zip(anchored_1, anchored_2, anchored_c):
        image_a, image_b, image_c = load(a), load(b), load(c)
        if not image_a.shape == image_b.shape == image_c.shape:
            print(f"FAIL: {a.name}: sizes differ {image_a.shape} / {image_b.shape} / {image_c.shape}")
            ok = False
            noises.append(0.0)
            shares.append(None)
            continue
        noises.append(changed_share(image_a, image_b))
        shares.append(changed_share(image_a, image_c))
        if diff_folder:
            amplified = np.clip(np.abs(image_a - image_c) * 4, 0, 255).astype(np.uint8)
            Image.fromarray(amplified).save(Path(diff_folder) / f"diff_{a.stem}_{c.stem}.png")
    print(f"threshold per capture = max(2 x its noise (base_1 vs base_2), {floor}%)")
    for a, c, noise, share in zip(anchored_1, anchored_c, noises, shares):
        if share is None:
            continue
        threshold = max(2.0 * noise, floor)
        good = share <= threshold
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'} {a.name} vs {c.name}: {share:.3f}% pixels > 8 (noise {noise:.3f}%, threshold {threshold:.3f}%)")
    if eyeball_c:
        print("eyeball (frame-number captures, not judged): " + ", ".join(str(shot) for shot in eyeball_c))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a", nargs="?")
    parser.add_argument("b", nargs="?")
    parser.add_argument("--diff", help="folder for the difference images")
    parser.add_argument("--tolerance", nargs=3, metavar=("BASE_1", "BASE_2", "CANDIDATE"), help="tolerance mode (see above)")
    parser.add_argument("--floor", type=float, default=0.1, help="tolerance mode: minimum threshold, in %% of pixels (default 0.1)")
    args = parser.parse_args()
    if args.tolerance:
        sys.exit(tolerance(*args.tolerance, args.floor, args.diff))
    if not args.a or not args.b:
        parser.error("give two run folders, or --tolerance BASE_1 BASE_2 CANDIDATE")
    shots_a = sorted(Path(args.a).glob("screenshot_*.png"))
    shots_b = sorted(Path(args.b).glob("screenshot_*.png"))
    if not shots_a or not shots_b:
        sys.exit("no screenshots in one of the folders")
    if len(shots_a) != len(shots_b):
        print(f"warning: {len(shots_a)} vs {len(shots_b)} screenshots; comparing the first {min(len(shots_a), len(shots_b))}")
    if args.diff:
        Path(args.diff).mkdir(parents=True, exist_ok=True)
    for a, b in zip(shots_a, shots_b):
        image_a, image_b = load(a), load(b)
        if image_a.shape != image_b.shape:
            print(f"{a.name} vs {b.name}: sizes differ {image_a.shape} vs {image_b.shape}")
            continue
        difference = np.abs(image_a - image_b)
        mean = difference.mean()
        changed = (difference.max(axis=2) > 8).mean() * 100.0
        print(f"{a.name} vs {b.name}: mean {mean:.2f}, {changed:.2f}% pixels > 8, max {int(difference.max())}")
        if args.diff:
            amplified = np.clip(difference * 4, 0, 255).astype(np.uint8)
            Image.fromarray(amplified).save(Path(args.diff) / f"diff_{a.stem}_{b.stem}.png")


if __name__ == "__main__":
    main()
