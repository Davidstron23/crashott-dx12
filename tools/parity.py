r"""Compares one game frame between Vulkan and Direct3D 12 (spec 003, runtime/gpu/parity.h).

    python tools/parity.py <profile> [--set key=value ...] [--replay-env API:KEY=VALUE ...]
                           [--replay-userdata API:folder ...] [--keep] [--no-build]

The profile must have `parity = [...]`: the frames to compare, as anchored times (CRASH_SCREENSHOT_TIMES).
The profile runs once with Vulkan, capturing those frames (run_game.py --parity); then each frame is
replayed on its own, twice with Vulkan and once with D3D12, always on a copy of the run's userdata, with
the passes of the image (upscaler, anti-aliasing, lighting) forced off. Per frame it reports:
  RF-2  the two Vulkan images are identical, byte for byte
  RF-3  Vulkan against D3D12: % of pixels with a channel > 8 (compare_runs.changed_share), against U = 0.1 %
  RF-4  when that fails, the first draws whose result differs (draws.csv: shaders, render target)
  RF-5  a frame is REJECTED, not compared, if it was not captured, a replay did not run the captured packets,
        or a replay used another renderer than the one asked for (D3D12 falling back to Vulkan)
--replay-env adds an environment variable to the replays of one API (d3d12:CRASH_D3D12_FAIL=device);
--replay-userdata copies a folder over that API's userdata copy (a shaderoverride). The captures (game
memory: ~120 MB each) are deleted when the frame passes, unless --keep. Exit code: 0 = every frame passes.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "perf"))
import run_game  # noqa: E402
from compare_runs import changed_share  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

THRESHOLD = 0.1  # U (P2): % of pixels with a channel that differs by more than 8
REPLAY_TIMEOUT = 300
# Passes the D3D12 renderer does not have yet (spec 004): off in both APIs.
PASSES_OFF = "graphics.upscaler=off;video.antialiasing=off;lighting.ssao=off;lighting.gi=false;lighting.contact_shadows=false;lighting.soft_shadows=false"
RUNS = [("vulkan", 1), ("vulkan", 2), ("d3d12", 1)]


def replay(run_dir, frame, api, number, extra_env, extra_userdata, background=False):
    """One replay; returns (problem or None). Moves its image and draws.csv to <api>_<number>.*"""
    info = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    userdata = run_dir / f"userdata_{api}"
    if not userdata.exists():
        shutil.copytree(run_dir / "userdata", userdata)
        for folder in extra_userdata.get(api, []):
            shutil.copytree(folder, userdata, dirs_exist_ok=True)
    args = list(info["args"])
    args[args.index("--userdata") + 1] = str(userdata)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("CRASH_")}
    env["CRASH_NO_HID"] = "1"
    env["CRASH_SETTINGS"] = ";".join(x for x in (info["env"].get("CRASH_SETTINGS"), PASSES_OFF) if x)
    for key, value in extra_env.get(api, []):
        env[key] = f"{env[key]};{value}" if key == "CRASH_SETTINGS" else value
    log = frame / f"{api}_{number}.log"
    with open(log, "wb") as out:
        process = subprocess.Popen(args + ["--renderer", api, "--parity-replay", str(frame)], stdout=out, stderr=subprocess.STDOUT, env=env)
        try:
            with run_game.window_away(process.pid, background):
                code = process.wait(timeout=REPLAY_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return f"{api} replay took over {REPLAY_TIMEOUT} s"
    text = log.read_text(encoding="utf-8", errors="replace")
    used = re.search(r"^\[parity\] renderer in use: (\w+)", text, re.MULTILINE)
    if code == 2:
        return "the replay did not run the captured packets: " + ([l for l in text.splitlines() if l.startswith("[parity]")] or ["?"])[-2]
    if code != 0 or not used:
        return f"{api} replay failed (exit {code}, see {log.name})"
    if used.group(1) != api:
        return f"asked for {api}, the replay used {used.group(1)}"
    for suffix in (".png", "_draws.csv"):
        shutil.move(frame / f"replay_{api}{suffix}", frame / f"{api}_{number}{suffix}")
    return None


def draws(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def first_differences(a, b, count=3):
    """RF-4: the first rows (draws and resolves) where the two replays left something different."""
    found = []
    for x, y in zip(a, b):
        same_reason = bool(x["reason"]) == bool(y["reason"])
        same_digest = "n/c" in (x["digest"], y["digest"]) or x["digest"] == y["digest"]
        if not (same_reason and same_digest):
            found.append(f"      draw {x['index']}: {x['type']} vs {x['vs']} ps {x['ps']} target {x['rt']}: "
                         f"vulkan '{x['reason'] or 'drawn'}' {x['digest']} / d3d12 '{y['reason'] or 'drawn'}' {y['digest']}")
            if len(found) == count:
                break
    return found


def compare(frame, name):
    """Prints the report of one frame; returns True if it passes."""
    images = {run: np.asarray(Image.open(frame / f"{run[0]}_{run[1]}.png").convert("RGBA")) for run in RUNS}
    v1, v2, d3 = (images[run] for run in RUNS)
    ok = True
    differing = int((v1 != v2).any(axis=2).sum()) if v1.shape == v2.shape else -1
    print(f"  {name} RF-2 vulkan x2: {'identical' if differing == 0 else f'{differing} pixels differ'}")
    ok &= differing == 0
    live = frame / "live.png"  # the frame as the running game drew it: only reported (the replay differs a little)
    if live.exists() and np.asarray(Image.open(live)).shape[:2] == v1.shape[:2]:
        print(f"  {name} (info) live game vs vulkan replay: {changed_share(np.asarray(Image.open(live).convert('RGB'), dtype=np.int16), v1[:, :, :3].astype(np.int16)):.3f}%")
    if v1.shape != d3.shape:
        print(f"  {name} RF-3 FAIL: sizes {v1.shape} / {d3.shape}")
        return False
    share = changed_share(v1[:, :, :3].astype(np.int16), d3[:, :, :3].astype(np.int16))
    good = share <= THRESHOLD
    print(f"  {name} RF-3 vulkan vs d3d12: {share:.3f}% of pixels differ by more than 8 (U = {THRESHOLD}%): {'pass' if good else 'FAIL'}")
    if not good:
        print(f"  {name} RF-4 first draws that differ:")
        found = first_differences(draws(frame / "vulkan_1_draws.csv"), draws(frame / "d3d12_1_draws.csv"))
        print("\n".join(found) if found else "      none by digest or discard (a target of another size, or a pass after the draws)")
    return ok and good


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Compares a frame of a profile between Vulkan and Direct3D 12.")
    parser.add_argument("profile")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="as run_game.py --set")
    parser.add_argument("--replay-env", action="append", default=[], metavar="API:KEY=VALUE")
    parser.add_argument("--replay-userdata", action="append", default=[], metavar="API:FOLDER")
    parser.add_argument("--keep", action="store_true", help="keep the captures after a pass")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--background", action="store_true", help="keep the game windows off screen and at the back (run_game.py --background)")
    args = parser.parse_args()

    extra_env, extra_userdata = {}, {}
    for item in args.replay_env:
        api, _, rest = item.partition(":")
        key, _, value = rest.partition("=")
        extra_env.setdefault(api, []).append((key, value))
    for item in args.replay_userdata:
        api, _, folder = item.partition(":")
        extra_userdata.setdefault(api, []).append(folder)

    config = run_game.load_config(run_game.CONFIG)
    profile = config["profiles"].get(args.profile)
    if not profile or not profile.get("parity"):
        sys.exit(f"{args.profile}: no such profile, or it has no `parity` frames in config/run.toml")
    overrides = dict(run_game.parse_override(s) for s in args.set)
    env = dict(overrides.get("env", profile.get("env", {})))
    env.setdefault("CRASH_NO_HID", "1")
    env["CRASH_SETTINGS"] = ";".join(x for x in (env.get("CRASH_SETTINGS"), PASSES_OFF) if x)
    overrides.update(env=env, renderer="vulkan", _parity=True)

    started = time.time()
    result = run_game.run_profile(config, args.profile, overrides, config.get("build", {}).get("auto", True) and not args.no_build,
                                  background=args.background)
    run_dir = Path(result["run_dir"])
    wanted = len(profile["parity"])
    print(f"== parity {args.profile}: capture done in {time.time() - started:.0f} s")
    all_ok = True
    for n in range(1, wanted + 1):
        frame, name = run_dir / f"parity_{n}", f"frame {n}/{wanted} ({profile['parity'][n - 1]})"
        problem = None if (frame / "memory.bin").exists() else "not captured (the event did not come before the timeout, or the game was not held: see run.log)"
        for api, number in RUNS:
            problem = problem or replay(run_dir, frame, api, number, extra_env, extra_userdata, args.background)
        if problem:
            print(f"  {name} REJECTED (RF-5): {problem}")
            all_ok = False
            continue
        passed = compare(frame, name)
        all_ok &= passed
        if passed and not args.keep:
            for heavy in ("memory.bin", "state.bin", "stream.bin"):
                (frame / heavy).unlink(missing_ok=True)
    if all_ok and not args.keep:
        for api in ("vulkan", "d3d12"):
            shutil.rmtree(run_dir / f"userdata_{api}", ignore_errors=True)
    print(f"== {'PASS' if all_ok else 'FAIL'} in {time.time() - started:.0f} s; images in {run_dir}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
