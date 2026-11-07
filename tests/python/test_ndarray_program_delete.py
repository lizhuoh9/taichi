# Added in the fork lizhuoh9/taichi on 2026-11-06 (see FORK.md): tests that an ndarray is deleted only on its program.
"""A ScalarNdarray is deleted only on the program that created it (FORK.md).  Taichi 1.7.4's ScalarNdarray.__del__
passed its pointer to the current program's delete_ndarray.  When Python frees an ndarray of an earlier program after
the next ti.init (for example a reference cycle that the cyclic collector frees later), that pointer is already freed,
and the current program erases whichever of its own ndarrays now sits at that address: the next kernel that takes the
live ndarray fails with "invalid DeviceAllocation".  A process that initialises Taichi again, such as a test process,
is exposed; a program that calls ti.init once is not.

The risks: an earlier program's ndarray deleted on the current program, a current ndarray no longer deleted (a leak),
and an ndarray that keeps its program's runtime alive.  Each check runs a script in a fresh interpreter on the CPU,
because it initialises Taichi twice and controls when the cyclic collector runs.
"""

import json
import os
import subprocess
import sys

CPU = "arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False, log_level=ti.WARN"


def run_script(directory, text):
    """Run `text` as a script file in a fresh interpreter; the process must exit 0.  Returns its stdout."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "check.py"
    script.write_text(text, encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", TI_SKIP_VERSION_CHECK="ON")
    done = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return done.stdout


def tagged(stdout, tag):
    """The JSON payload of the one stdout line that starts with `tag`."""
    (line,) = [line for line in stdout.splitlines() if line.startswith(tag + " ")]
    return json.loads(line[len(tag) + 1 :])


DELETE_CALLS = f"""
import gc
import json
import weakref

import taichi as ti
from taichi.lang import impl


class Recorder:
    # Stands in for the current program while ndarrays are freed: records each delete_ndarray call, then makes it.
    def __init__(self, prog):
        self.prog, self.deleted = prog, []

    def delete_ndarray(self, arr):
        self.deleted.append(arr)
        self.prog.delete_ndarray(arr)

    def __getattr__(self, name):
        return getattr(self.prog, name)


ti.init({CPU})
earlier = ti.ndarray(ti.f32, shape=8)
earlier_arr, earlier_runtime = earlier.arr, weakref.ref(impl.get_runtime())
ti.reset()
ti.init({CPU})
gc.collect()
print("EARLIER_RUNTIME_ALIVE " + json.dumps(earlier_runtime() is not None))
current = ti.ndarray(ti.f32, shape=8)
current_arr = current.arr
runtime = impl.get_runtime()
recorder = Recorder(runtime.prog)
runtime.prog = recorder
try:
    del earlier, current
finally:
    runtime.prog = recorder.prog
names = ["earlier" if arr is earlier_arr else "current" if arr is current_arr else "other" for arr in recorder.deleted]
print("DELETED " + json.dumps(names))
"""


def test_an_ndarray_is_deleted_on_its_own_program_only(tmp_path):
    """Freeing an earlier program's ndarray makes no delete_ndarray call on the current program (Taichi 1.7.4 made
    one, with a pointer that program never owned), a current ndarray is still deleted exactly once, and an ndarray
    does not keep its program's runtime alive (the runtime is held weakly)."""
    stdout = run_script(tmp_path, DELETE_CALLS)
    assert tagged(stdout, "DELETED") == ["current"]
    assert tagged(stdout, "EARLIER_RUNTIME_ALIVE") is False


STALE_CYCLES = f"""
import gc
import json

import taichi as ti

N, ROUNDS = 256, 4


@ti.kernel
def write(a: ti.types.ndarray(), value: ti.f32):
    for i in a:
        a[i] = value


broken = []
gc.disable()  # the cycles below survive until the explicit collect
for round_index in range(ROUNDS):
    ti.init({CPU})
    for _ in range(N):
        holder = {{"arr": ti.ndarray(ti.f32, shape=8)}}
        holder["self"] = holder  # a reference cycle: only the cyclic collector frees it
    del holder
    ti.reset()
    ti.init({CPU})
    current = [ti.ndarray(ti.f32, shape=8) for _ in range(N)]
    gc.collect()  # the earlier program's ndarrays are freed now, while the current program owns its own
    for index, arr in enumerate(current):
        try:
            write(arr, float(index))
            if arr.to_numpy().tolist() != [float(index)] * 8:
                broken.append([round_index, index, "values"])
        except RuntimeError as error:
            broken.append([round_index, index, str(error).splitlines()[0][:200]])
    del current
gc.enable()
print("BROKEN " + json.dumps(broken))
"""


def test_freeing_an_earlier_programs_ndarrays_leaves_the_current_ones_intact(tmp_path):
    """The failure itself, as in Taichi's test_ad_ndarray.py under a re-initialising test process: 256 ndarrays of
    an earlier program, kept by reference cycles, are freed after the next ti.init while the current program owns 256
    ndarrays of the same size, in 4 rounds.  Every current ndarray must still take a kernel and hold its values.  On
    Taichi 1.7.4 some of them are freed, on the official wheel too, but not in every round: whether a stale address
    meets a live ndarray depends on the allocator (in development runs: 0 to 8 of 256 in one round)."""
    stdout = run_script(tmp_path, STALE_CYCLES)
    assert tagged(stdout, "BROKEN") == []
