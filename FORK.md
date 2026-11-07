# The fork lizhuoh9/taichi

This repository is an unofficial fork of [taichi-dev/taichi](https://github.com/taichi-dev/taichi), not affiliated with the
Taichi project. Taichi is licensed under the Apache License 2.0 (`LICENSE`, unchanged), and so are the fork's changes.

## Base

- Branch `main`, built on the upstream tag `v1.7.4` (commit `b4b956fdaa67fd3e198527fd607053ecb9bf4aa2`, the commit the official
  1.7.4 wheel reports). The branch `master` stays upstream's.
- The fork changes Python files only. The C++ core is the release's: the fork's wheel carries the official wheel's binaries byte
  for byte, so `ti.__version__` stays `(1, 7, 4)`. The fork's own version is `taichi._fork.VERSION`.

## Changes

Every code file the fork changes or adds carries a header comment with the date and a one-line summary (Apache-2.0,
section 4b).

1. Launch fast paths in `python/taichi/lang/kernel_impl.py`, on by default. Taichi's host time per kernel call is about 9 to 52
   microseconds on the CPU backend of an AMD Ryzen 9 5950X, much of it Python that repeats at every call, and it bounds the step of a
   program that launches many small kernels per step. Each fast path returns what Taichi 1.7.4 returns and raises what it raises, and hands
   every case it does not cover to Taichi's code:
   - `TaichiCallableTemplateMapper.extract_arg` keys a template argument by a decision made once per type (the
     argument itself, a weak reference to it, or Taichi's per-instance branches);
   - `_process_args` returns a call's positional arguments when they fill every parameter;
   - `Kernel.launch_kernel` launches a kernel whose non-template parameters are all primitive scalars (at most 64)
     with a plan made at its first launch, and keeps each compiled kernel's `CompiledKernelData` per program instead
     of asking the program for it at every launch. Taichi 1.7.4 launches a compiled kernel with the data of its first
     compilation, also after a compile-config write such as autodiff's `opt_level` 0 -> 1, so the results stay bit
     for bit.
2. The switch `TI_LAUNCH_FASTPATH`, read once when `taichi` is imported: unset or `1` turns the fast paths on, `0`
   turns them off (A/B timing within one installation), and any other value stops the import with a `ValueError`.
3. The marker `taichi._fork` (`python/taichi/_fork.py`, imported by `python/taichi/__init__.py`): `NAME`
   (`"lizhuoh9/taichi"`), `VERSION` (`"1.7.4+1"`), `BASE` (`"v1.7.4"`) and `LAUNCH_FASTPATH`. A program that
   needs the fork reads it; stock Taichi has no `taichi._fork`. The underscore keeps Taichi's public namespace as
   released.
4. Tests: `tests/python/test_launch_fastpath.py` compares every argument kind, value, return value and error with the
   fast paths on and off, the kept data's lifetime across `ti.init`, `ti.reset` and `SNodeTree.destroy`, the
   compile-config write and the switch.
5. `misc/fork_repack_wheel.py` builds the fork's wheel (below).

## Versions

- `1.7.4+1`: changes 1 to 5.

## Wheels

The fork changes Python files only, so its wheel is the official wheel of its base release, repacked:

```bash
pip download taichi==1.7.4 --no-deps --only-binary=:all: --platform manylinux_2_27_x86_64 \
    --python-version 3.10 --implementation cp --abi cp310 -d stock/
git archive v1.7.4 python/taichi | tar -x -C base/
git archive main python/taichi | tar -x -C fork/
python misc/fork_repack_wheel.py stock/taichi-1.7.4-cp310-cp310-manylinux_2_27_x86_64.whl base fork out \
    --sha256 42ffa0ba20b19e8695894cc4796ebefaed11cc10a1ac3704bd48a9ddfd54436e
```

The script puts the fork's changed Python files into the official wheel, sets the version to `taichi._fork.VERSION`
in `METADATA` and in the dist-info name, and regenerates `RECORD` with `wheel unpack` / `wheel pack`. It refuses any
difference between the two wheels other than the changed Python files, `METADATA`'s version line and `RECORD`, so
the C++ binaries stay byte-identical; `repack_files.txt` lists the sha256 of every file in both wheels. Every entry
of the new wheel carries the official wheel's newest entry time, so the same inputs give the same wheel byte for
byte. The sha256 above is PyPI's for the official CPython 3.10 Linux wheel of 1.7.4.

Releases of the fork attach the wheel for Linux x86-64 and CPython 3.10, the only combination tested. The script can make the
wheels of the other official platforms and Python versions; they are untested.

Install it with `pip install taichi-1.7.4+1-cp310-cp310-manylinux_2_27_x86_64.whl`, into the environment first. A later
`pip install taichi` or `pip install -r` with `taichi==1.7.4` keeps it (a local label satisfies that pin). What replaces it is an
explicit request for the official wheel, such as `pip install --force-reinstall taichi==1.7.4`; afterwards `taichi._fork` is gone.
