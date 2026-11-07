"""Sets up a clone of this fork for development from an installed copy of CrashRecomp, instead of
building everything again from the game dump (tools/setup.py).

    python tools/dev_setup.py                         # the default install folder
    python tools/dev_setup.py --install "D:/Games/Crash of the Titans Definitive Edition"
    python tools/dev_setup.py --no-build --no-dist    # only the links, copies and hooks

What it does, in order (each step skips what is already there):
1. Links the game-derived folders of the installation's project folder (<install>\\CrashRecomp) into
   the clone as directory junctions: private\\, generated\\, data\\, toolchain\\ and
   tools\\XenonRecomp\\. Nothing is copied, and git ignores all of them.
2. Copies the third-party binaries and SDK headers the build needs (DLSS, DXC, FFmpeg, FSR) and the
   subtitle texts, which are not in git. They are copied, not linked: the D3D12 work changes them.
   dxil.dll (the DXIL validator the Direct3D 12 renderer needs) is downloaded from the official DXC
   release when the installation's copy of DXC does not have it (an older installation).
3. Writes the game folder (<install>\\game) into config/local.toml, for tools/run_game.py.
4. Turns on the repository's git hooks (.githooks: no game files, no direct push to main).
5. Configures and builds build\\ with the bundled toolchain (tools/buildenv.py).
6. Updates dist\\ (tools/make_dist.py), the test copy to play the new build from (UAT): its own exe,
   DLLs and userdata\\ (saves, settings), reading the game from <install>\\game. The installation
   itself is only read, never written.

Never delete the junctions with a tool that follows links (rm -rf, git clean -x, shutil.rmtree on old
Pythons): it would delete the installation's files. Remove a junction with `rmdir <folder>` (cmd).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import buildenv  # noqa: E402

ROOT = buildenv.ROOT
DEFAULT_INSTALL = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Crash of the Titans Definitive Edition"

# Linked from <install>\CrashRecomp: made from the game (or large and unchanged by this fork).
LINKS = ["private", "generated", "data", "toolchain", "tools/XenonRecomp"]
# Copied from <install>\CrashRecomp: third-party files the build uses.
COPIES = ["thirdparty/dlss/bin", "thirdparty/dxc/bin", "thirdparty/ffmpeg/bin", "thirdparty/ffmpeg/include",
          "thirdparty/fsr/bin", "thirdparty/fsr/include"]


def step(text):
    print(f"\n== {text}", flush=True)


def check_install(install):
    """Raises SystemExit unless `install` is a CrashRecomp installation with its project folder."""
    project = install / "CrashRecomp"
    if not (install / "game" / "default.xex").is_file():
        raise SystemExit(f"{install} has no game\\default.xex: pass the installation folder with --install")
    if not any((project / "generated" / "ppc").glob("*.cpp")):
        raise SystemExit(f"{project} has no generated\\ppc\\*.cpp: the installation was not built from its project folder")
    return project


def link_folders(project):
    for name in LINKS:
        target = project / name
        link = ROOT / name
        if os.path.lexists(link):
            print(f"  {name}: already there, kept")
            continue
        if not target.is_dir():
            print(f"  {name}: not in the installation, skipped")
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        # A junction needs no administrator rights or developer mode, unlike a symbolic link.
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, stdout=subprocess.DEVNULL)
        print(f"  {name} -> {target}")


def copy_third_party(project):
    for name in COPIES:
        source = project / name
        target = ROOT / name
        if target.exists():
            print(f"  {name}: already there, kept")
        elif source.is_dir():
            shutil.copytree(source, target)
            print(f"  {name}: copied")
        else:
            print(f"  {name}: not in the installation, skipped (that feature will be off)")
    for source in (project / "languages").glob("*/subtitles.txt"):
        target = ROOT / source.relative_to(project)
        if not target.exists():
            shutil.copy2(source, target)
            print(f"  {target.relative_to(ROOT).as_posix()}: copied")


def fetch_dxil():
    """dxil.dll, from the official DXC release (SHA-256 checked by setup.py), if the copied DXC lacks it."""
    if (ROOT / "thirdparty" / "dxc" / "bin" / "dxil.dll").is_file():
        print("  thirdparty/dxc/bin/dxil.dll: already there, kept")
        return
    import setup as crash_setup  # tools/setup.py, the same folder
    crash_setup.fetch_dxc()
    print("  thirdparty/dxc/bin/dxil.dll: downloaded")


def set_game_folder(install):
    game = install / "game"
    current = buildenv.local().get("paths", {}).get("game")
    if current:
        print(f"  config/local.toml already has [paths] game = {current}, kept")
    else:
        buildenv.write_local(game=game)
        print(f"  config/local.toml: [paths] game = {game.as_posix()}")


def enable_hooks():
    subprocess.run(["git", "-C", str(ROOT), "config", "core.hooksPath", ".githooks"], check=True)
    print("  core.hooksPath = .githooks")


def build():
    env = buildenv.build_env()
    for command in buildenv.build_commands(ROOT, ROOT / "build", extra=["-DCRASH_INSTALLER_ONLY=OFF"]):
        command = [str(part) for part in command]
        # Windows looks the program up in this process's PATH, not env's: find cmake in the toolchain's.
        command[0] = shutil.which(command[0], path=env["PATH"]) or command[0]
        subprocess.run(command, check=True, env=env)


def make_dist(install):
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_dist.py"), "--game", str(install / "game")], check=True)


def main():
    parser = argparse.ArgumentParser(description="Sets up this clone for development from an installed CrashRecomp.")
    parser.add_argument("--install", type=Path, default=DEFAULT_INSTALL,
                        help="the installation folder (default: %(default)s)")
    parser.add_argument("--no-build", action="store_true", help="skip configuring and building build\\")
    parser.add_argument("--no-dist", action="store_true", help="skip updating dist\\ (the test copy)")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("dev_setup.py is for Windows (the D3D12 renderer is Windows only)")

    install = args.install.resolve()
    project = check_install(install)
    print(f"installation: {install}")

    step("Linking the game-derived folders")
    link_folders(project)
    step("Copying the third-party files")
    copy_third_party(project)
    step("DXIL validator (dxil.dll) for Direct3D 12")
    fetch_dxil()
    step("Game folder for the tests")
    set_game_folder(install)
    step("Git hooks")
    enable_hooks()
    if not args.no_build:
        step("Building build\\")
        build()
    if not args.no_dist:
        step("Updating dist\\ (the test copy)")
        make_dist(install)
    print("\nDone. Play the new build with dist\\CrashLauncher.exe; tests: python tools/run_game.py --list")


if __name__ == "__main__":
    main()
