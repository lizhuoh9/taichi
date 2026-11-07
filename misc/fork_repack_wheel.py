# Added in the fork lizhuoh9/taichi on 2026-11-06 (see FORK.md): builds the fork's wheel from the official wheel.
"""Repack an official Taichi wheel into the fork's wheel (FORK.md, "Wheels").

The fork changes Python files only, so its wheel is the official wheel of its base release with the fork's changed
Python files put in, the version set to taichi._fork.VERSION (METADATA and the dist-info name) and RECORD regenerated
by `wheel unpack` / `wheel pack`.  The C++ binaries and every other file stay byte-identical; the script checks that.
Every entry of the new wheel carries the official wheel's newest entry time, so the same inputs give the same wheel.

    python misc/fork_repack_wheel.py STOCK_WHEEL BASE_TREE FORK_TREE OUT_DIR [--sha256 DIGEST]

STOCK_WHEEL is the official wheel (checked against DIGEST when given), BASE_TREE and FORK_TREE are exports (git
archive) of the base tag and of the fork's commit, and OUT_DIR is a new directory for the wheel and its report.  The
changed files are the .py files under python/taichi/ whose bytes differ between the two trees, or that only the fork
has.  It refuses: a non-.py change under python/taichi/, a file the fork removed, a changed file whose copy in the
wheel is not the base tree's, and any difference between the two wheels other than the changed files, METADATA's
Version line and RECORD.  It needs the `wheel` package (python -m wheel).
"""

import argparse
import ast
import calendar
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

PACKAGE = pathlib.PurePosixPath("python/taichi")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def tree_files(root):
    """{path relative to root/python/taichi: bytes} for every file there."""
    base = pathlib.Path(root, PACKAGE)
    if not base.is_dir():
        raise SystemExit(f"no {PACKAGE} in {root}")
    return {path.relative_to(base).as_posix(): path.read_bytes() for path in base.rglob("*") if path.is_file()}


def changed_files(base_tree, fork_tree):
    """The fork's changed .py files under python/taichi/ (module docstring), with their bytes."""
    base, fork = tree_files(base_tree), tree_files(fork_tree)
    removed = sorted(set(base) - set(fork))
    changed = {path: data for path, data in fork.items() if base.get(path) != data}
    not_python = sorted(path for path in changed if not path.endswith(".py"))
    if removed or not_python:
        raise SystemExit(f"refused: removed {removed}, changed files that are not .py {not_python}")
    return changed, base


def fork_version(fork_tree):
    """taichi._fork.VERSION, read from the source without importing it."""
    module = ast.parse(pathlib.Path(fork_tree, PACKAGE, "_fork.py").read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["VERSION"]:
            return ast.literal_eval(node.value)
    raise SystemExit("no VERSION in python/taichi/_fork.py")


def source_date_epoch(path):
    """The newest entry time of a wheel, in seconds since the epoch: `wheel pack` stamps every entry of the new wheel
    with it (SOURCE_DATE_EPOCH), so the same inputs give the same wheel, byte for byte."""
    with zipfile.ZipFile(path) as wheel:
        return max(calendar.timegm(info.date_time + (0, 0, 0)) for info in wheel.infolist())


def wheel_listing(path):
    """{entry name: (sha256, external_attr)} of a wheel."""
    with zipfile.ZipFile(path) as wheel:
        return {info.filename: (sha256(wheel.read(info)), info.external_attr) for info in wheel.infolist()}


def compare(stock, fork, changed, stock_version, version):
    """The checks of the module docstring on two wheel listings; returns (report lines, problems)."""
    old_info, new_info = f"taichi-{stock_version}.dist-info/", f"taichi-{version}.dist-info/"
    renamed = {name.replace(old_info, new_info, 1): value for name, value in stock.items()}
    expected_changes = {"taichi/" + path for path in changed} | {new_info + "METADATA", new_info + "RECORD"}
    lines, problems = [], []
    for name in sorted(set(renamed) | set(fork)):
        old, new = renamed.get(name), fork.get(name)
        state = "equal" if old == new else ("changed" if name in expected_changes else "DIFFERENT")
        lines.append(f"{state:9} {old[0] if old else '-':64} {new[0] if new else '-':64} {name}")
        if state == "DIFFERENT":
            problems.append(name)
        if state == "changed" and old and new and old[1] != new[1]:
            problems.append(name + " (permissions)")
    return lines, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("stock_wheel")
    parser.add_argument("base_tree")
    parser.add_argument("fork_tree")
    parser.add_argument("out_dir")
    parser.add_argument("--sha256", help="the official wheel's published sha256")
    args = parser.parse_args()
    stock_path, out = pathlib.Path(args.stock_wheel), pathlib.Path(args.out_dir)
    if out.exists():
        raise SystemExit(f"refused: {out} exists")
    stock_digest = sha256(stock_path.read_bytes())
    if args.sha256 and stock_digest != args.sha256:
        raise SystemExit(f"refused: {stock_path} has sha256 {stock_digest}, not {args.sha256}")
    changed, base = changed_files(args.base_tree, args.fork_tree)
    version = fork_version(args.fork_tree)
    stock_version = stock_path.name.split("-")[1]
    if version.split("+")[0] != stock_version:
        raise SystemExit(f"refused: the fork's version {version} is not a local version of {stock_version}")
    out.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as scratch:
        subprocess.run([sys.executable, "-m", "wheel", "unpack", str(stock_path), "-d", scratch], check=True)
        (unpacked,) = pathlib.Path(scratch).iterdir()
        for path, data in sorted(changed.items()):
            target = unpacked / "taichi" / path
            if path in base and target.read_bytes() != base[path]:
                raise SystemExit(f"refused: the wheel's taichi/{path} is not the base tree's")
            target.write_bytes(data)
        info = unpacked / f"taichi-{stock_version}.dist-info"
        metadata = (info / "METADATA").read_text(encoding="utf-8").split("\n")
        versions = [i for i, line in enumerate(metadata) if line == f"Version: {stock_version}"]
        if len(versions) != 1:
            raise SystemExit(f"refused: METADATA has {len(versions)} lines 'Version: {stock_version}'")
        metadata[versions[0]] = f"Version: {version}"
        (info / "METADATA").write_bytes("\n".join(metadata).encode("utf-8"))
        info.rename(unpacked / f"taichi-{version}.dist-info")
        env = dict(os.environ, SOURCE_DATE_EPOCH=str(source_date_epoch(stock_path)))
        subprocess.run([sys.executable, "-m", "wheel", "pack", str(unpacked), "-d", str(out)], check=True, env=env)
    (fork_path,) = out.glob("*.whl")
    lines, problems = compare(wheel_listing(stock_path), wheel_listing(fork_path), changed, stock_version, version)
    report = dict(
        stock_wheel=stock_path.name,
        stock_sha256=stock_digest,
        fork_wheel=fork_path.name,
        fork_sha256=sha256(fork_path.read_bytes()),
        version=version,
        changed_files=sorted("taichi/" + path for path in changed),
        problems=problems,
    )
    (out / "repack_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "repack_files.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if problems:
        shutil.move(str(fork_path), str(fork_path) + ".refused")
        raise SystemExit(f"refused: the wheels differ beyond the fork's changes: {problems}")


if __name__ == "__main__":
    main()
