# Added in the fork lizhuoh9/taichi on 2026-11-07 (see FORK.md): tests of fast path e, the leaner kernel call.
"""Fast path e of taichi.lang.kernel_impl (FORK.md change 8) sends a primal kernel called with one positional argument
per parameter, outside ti.ad.FwdMode and ti.ad.Tape, straight to ensure_compiled and launch_kernel, and fast path c's
launch is written inside launch_kernel. The risks: a call inside ti.ad.Tape or ti.ad.FwdMode, or of a gradient kernel,
that skips what Taichi 1.7.4 does there (the tape's record, forward mode's kernel swap and recovery); a call with
keywords, a missing or an extra argument or inspect.Parameter.empty that loses 1.7.4's error; a scalar launch with
another value, conversion or error; a data-oriented kernel that is called differently; calls after ti.reset and
ti.init; a launch that a wrapper on Kernel.__call__ no longer counts (a downstream solver's phase timer counts the
launches per phase with such a wrapper).

One script makes every observation in a fresh interpreter on the CPU, with TI_LAUNCH_FASTPATH unset (on) and 0 (off,
Taichi 1.7.4's code); the observations must be equal.
"""

import json
import os
import subprocess
import sys

import pytest

CPU = "arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False, log_level=ti.WARN"

SCENARIOS = r"""
import inspect
import json
import re

import numpy as np
import taichi as ti
from taichi.lang import kernel_impl

ti.init(CPU_ARGUMENTS)
OBSERVED = {}


def clean(text):
    return re.sub(r" at 0x[0-9a-fA-F]+", " at 0x?", text)


def see(name, function):
    assert name not in OBSERVED, name
    try:
        OBSERVED[name] = ["value", clean(repr(function()))]
    except Exception as error:  # an error is an observation
        OBSERVED[name] = ["error", type(error).__name__, clean(str(error))]


x = ti.field(ti.f32, shape=4, needs_grad=True, needs_dual=True)
y = ti.field(ti.f32, shape=4, needs_grad=True, needs_dual=True)
loss = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
x.from_numpy(np.arange(4, dtype=np.float32))


@ti.kernel
def scale(src: ti.template(), dst: ti.template(), factor: ti.f32):
    for i in src:
        dst[i] = src[i] * factor


@ti.kernel
def total(src: ti.template()):
    for i in src:
        loss[None] += src[i] ** 2


@ti.kernel
def answer(a: ti.i32, b: ti.f32) -> ti.f32:
    return a * b


@ti.kernel
def pair(a: ti.u8, b: ti.i64) -> ti.types.vector(2, ti.i64):
    return ti.Vector([a, b])


@ti.data_oriented
class Holder:
    def __init__(self, n):
        self.data = ti.field(ti.f32, shape=n)

    @ti.kernel
    def fill(self, value: ti.f32):
        for i in self.data:
            self.data[i] = value + i

    @ti.kernel
    def total(self) -> ti.f32:
        s = 0.0
        for i in self.data:
            s += self.data[i]
        return s


see("positional", lambda: (scale(x, y, 2.0), y.to_numpy().tolist())[1])
see("keywords", lambda: (scale(src=x, dst=y, factor=3.0), y.to_numpy().tolist())[1])
see("positional and keyword", lambda: (scale(x, y, factor=4.0), y.to_numpy().tolist())[1])
see("missing argument", lambda: scale(x, y))
see("extra argument", lambda: scale(x, y, 1.0, 2.0))
see("inspect.Parameter.empty", lambda: scale(x, y, inspect.Parameter.empty))
see("unknown keyword", lambda: scale(x, y, factr=1.0))
see("keyword twice", lambda: scale(x, y, 1.0, factor=2.0))
see("return value", lambda: answer(3, 0.5))
see("return value, keywords", lambda: answer(b=0.25, a=8))
see("bad scalar", lambda: answer("3", 0.5))
see("float for an int", lambda: answer(3.0, 0.5))
see("numpy scalars", lambda: answer(np.int32(5), np.float64(0.5)))
see("unsigned and wide", lambda: pair(255, -(2 ** 40)).to_numpy().tolist())
see("negative for unsigned", lambda: pair(-1, 0).to_numpy().tolist())

with ti.ad.Tape(loss=loss) as tape:
    total(x)
    scale(x, y, 0.5)
see("tape calls", lambda: [(call[0].func.__name__, type(call[1]).__name__, len(call[1])) for call in tape.calls])
see("tape modes", lambda: [str(mode) for mode in tape.modes])
see("tape gradient", lambda: x.grad.to_numpy().tolist())
see("tape loss", lambda: loss[None])
see("modes after the tape", lambda: [str(k._primal.autodiff_mode) for k in (total, scale)])

with ti.ad.FwdMode(loss=loss, param=x, seed=[0.0, 1.0, 0.0, 0.0]) as forward:
    total(x)
see("forward calls", lambda: [call.func.__name__ for call in forward.calls])
see("forward dual", lambda: loss.dual[None])
see("modes after forward mode", lambda: [str(k._primal.autodiff_mode) for k in (total, scale)])

loss.grad[None] = 1.0
x.grad.fill(0.0)
see("gradient kernel", lambda: (total.grad(x), x.grad.to_numpy().tolist())[1])
see("primal after its gradient", lambda: (total(x), loss[None])[1])

x64 = ti.field(ti.f64, shape=4, needs_grad=True)
loss64 = ti.field(ti.f64, shape=(), needs_grad=True)
x64.from_numpy(np.linspace(0.5, 2.0, 4))


@ti.kernel
def cubes(src: ti.template()):
    for i in src:
        loss64[None] += src[i] ** 3


def checked_tape():
    # Tape.__exit__ clears the tape before its gradient check, which calls the recorded primal kernels again
    np.random.seed(0)
    with ti.ad.Tape(loss=loss64, grad_check=[x64]) as checked:
        cubes(x64)
    return [x64.grad.to_numpy().tolist(), loss64[None], [call[0].func.__name__ for call in checked.calls]]


def validated_tape():
    with ti.ad.Tape(loss=loss, validation=True) as validated:
        total(x)
    return [x.grad.to_numpy().tolist(), loss[None], [str(mode) for mode in validated.modes]]


see("tape with a gradient check", checked_tape)
see("tape with validation", validated_tape)
see("mode after the validated tape", lambda: str(total._primal.autodiff_mode))

h = Holder(3)
see("data-oriented call", lambda: (h.fill(2.0), h.data.to_numpy().tolist())[1])
see("data-oriented keywords", lambda: (h.fill(value=3.0), h.data.to_numpy().tolist())[1])
see("data-oriented missing argument", lambda: h.fill())
see("data-oriented extra argument", lambda: h.fill(1.0, 2.0))
see("data-oriented return value", lambda: h.total())
see("data-oriented gradient attribute", lambda: type(h.fill.grad).__name__)

ti.reset()
ti.init(CPU_ARGUMENTS)
z = ti.field(ti.f32, shape=4)
w = ti.field(ti.f32, shape=4)
z.from_numpy(np.ones(4, dtype=np.float32))
see("after ti.init, new fields", lambda: (scale(z, w, 2.0), w.to_numpy().tolist())[1])
see("after ti.init, return value", lambda: answer(2, 0.25))
h2 = Holder(2)
see("after ti.init, data-oriented", lambda: (h2.fill(1.0), h2.total())[1])

# A wrapper on Kernel.__call__ as a downstream solver's phase timer installs and removes it, with a second count on
# launch_kernel: every launch passes the wrapper, and a refused call passes it without a launch.
CALLED, LAUNCHED = [], []
unwrapped_call, unwrapped_launch = kernel_impl.Kernel.__call__, kernel_impl.Kernel.launch_kernel


def counted_call(kernel, *args, **kwargs):
    CALLED.append(kernel.func.__name__)
    return unwrapped_call(kernel, *args, **kwargs)


def counted_launch(kernel, t_kernel, *args):
    LAUNCHED.append(kernel.func.__name__)
    return unwrapped_launch(kernel, t_kernel, *args)


def counted_section():
    scale(z, w, 1.0)  # compiled already
    scale(z, w, 2.0)
    scale(w, z, 1.0)  # a new kernel instance: it compiles here
    scale(z, w, factor=3.0)
    answer(2, 0.25)
    answer(a=2, b=0.5)
    h2.fill(1.0)
    h2.fill(value=2.0)
    h2.total()
    bound = h2.total
    bound()
    bound()
    try:
        scale(z, w)
    except ti.TaichiSyntaxError:
        pass
    return h2.data.to_numpy().tolist()  # a kernel of Taichi's own library


kernel_impl.Kernel.__call__, kernel_impl.Kernel.launch_kernel = counted_call, counted_launch
see("under a wrapper on Kernel.__call__", counted_section)
see("the wrapper's count", lambda: list(CALLED))
see("launches under it", lambda: list(LAUNCHED))
kernel_impl.Kernel.__call__ = unwrapped_call
see("a call after the wrapper is taken off", lambda: answer(4, 0.25))
see("the wrapper's count after that", lambda: len(CALLED))
see("launches after that", lambda: LAUNCHED[-1])
kernel_impl.Kernel.launch_kernel = unwrapped_launch
print("COUNTED " + json.dumps([CALLED, LAUNCHED]))
print("OBSERVED " + json.dumps(OBSERVED, sort_keys=True))
""".replace(
    "CPU_ARGUMENTS", CPU
)


def run_script(directory, text, switch):
    """Run `text` as a script file in a fresh interpreter; `switch` sets TI_LAUNCH_FASTPATH ("unset" removes it).
    Returns the finished process, which must exit 0; its stdout is kept beside the script."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "check.py"
    script.write_text(text, encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", TI_SKIP_VERSION_CHECK="ON")
    if switch == "unset":
        env.pop("TI_LAUNCH_FASTPATH", None)
    else:
        env["TI_LAUNCH_FASTPATH"] = switch
    done = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    (directory / "stdout.txt").write_text(done.stdout, encoding="utf-8")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return done


def tagged(done, tag):
    """The JSON payload of the one stdout line that starts with `tag`."""
    (line,) = [line for line in done.stdout.splitlines() if line.startswith(tag + " ")]
    return json.loads(line[len(tag) + 1 :])


@pytest.fixture(scope="module", name="runs")
def runs_on_and_off(tmp_path_factory):
    """The scenarios with fast path e on (TI_LAUNCH_FASTPATH unset) and off (0, Taichi 1.7.4's code)."""
    return (
        run_script(tmp_path_factory.mktemp("on"), SCENARIOS, "unset"),
        run_script(tmp_path_factory.mktemp("off"), SCENARIOS, "0"),
    )


def test_every_call_value_error_tape_and_mode_is_taichi_1_7_4s(runs):
    """Every observation is equal on and off: values and errors of calls with positional and keyword arguments, a
    missing or an extra argument, inspect.Parameter.empty, scalars of every kind and their errors; the tape's record
    (kernels, argument lists, modes) and gradient, forward mode's kernels and dual, the modes after both, a gradient
    kernel and the primal after it, a tape with a gradient check (whose calls of the primal kernels come after the
    tape is cleared) and one with validation; data-oriented calls, their errors and gradient attribute; calls after
    ti.reset and ti.init."""
    on, off = (tagged(run, "OBSERVED") for run in runs)
    assert set(on) == set(off)
    differing = {name: (on[name], off[name]) for name in on if on[name] != off[name]}
    assert not differing, differing
    assert on["tape calls"] == ["value", "[('total', 'list', 1), ('scale', 'list', 3)]"]
    assert on["modes after forward mode"] == ["value", "['AutodiffMode.NONE', 'AutodiffMode.NONE']"]
    assert on["missing argument"][0] == "error"
    assert on["tape with a gradient check"][0] == "value", on["tape with a gradient check"]


def test_a_wrapper_on_kernel_call_counts_every_launch(runs):
    """A downstream solver's phase timer counts the launches per phase with a wrapper on Kernel.__call__: on and off
    the wrapper sees every launch, of plain and data-oriented kernels, with positional and
    keyword arguments, through a kept bound kernel, and of Taichi's own library kernels; a refused call passes it
    without a launch, and the kernels run as before once it is taken off."""
    on, off = (tagged(run, "COUNTED") for run in runs)
    assert on == off
    called, launched = on
    assert called[:12] == ["scale"] * 4 + ["answer"] * 2 + ["fill"] * 2 + ["total"] * 3 + ["scale"]
    # the refused call (the twelfth) passes the wrapper and launches nothing; every other call launches once
    assert launched == called[:11] + called[12:] + ["answer"]
