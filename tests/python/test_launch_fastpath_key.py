# Added in the fork lizhuoh9/taichi on 2026-11-07 (see FORK.md): tests of fast path f, the template key reused.
"""Fast path f of taichi.lang.kernel_impl (FORK.md change 9) reuses a callable's last template key while every argument
that the key holds, or holds a weak reference to, is the same object. The risks: a key reused for arguments that get
another key (a different object, an object that died and the object after it, None after a weakly keyed object died,
an argument keyed by its state such as an array, an argument that Taichi keys per instance such as a tuple), a reused
key that keeps an object alive, and keys or kernel instances other than Taichi 1.7.4's, also after ti.reset and
ti.init. A downstream solver's patterns are among them: data-oriented objects called in turn, a data-oriented
argument whose field is replaced, equal bool and int template arguments.

One script makes every observation in a fresh interpreter on the CPU, with TI_LAUNCH_FASTPATH unset (on) and 0 (off,
Taichi 1.7.4's code); the observations must be equal.  The script also counts the key extractions of each kernel under
test, which only fast path f saves: off, every call extracts its key; on, a call whose key arguments are the objects
of the kernel's last call extracts none.
"""

import json
import os
import subprocess
import sys

import pytest

CPU = "arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False, log_level=ti.WARN"

SCENARIOS = r"""
import gc
import json
import os
import re
import weakref

import numpy as np
import taichi as ti
from taichi.lang import kernel_impl

ti.init(CPU_ARGUMENTS)
OBSERVED = {}
EXTRACTED = {}
COUNTS = {}
MAPPERS = {}


HERE = os.path.dirname(os.path.abspath(__file__))


def clean(text):
    return re.sub(r" at 0x[0-9a-fA-F]+", " at 0x?", text.replace(HERE, "<here>"))


def see(name, function):
    assert name not in OBSERVED, name
    try:
        OBSERVED[name] = ["value", clean(repr(function()))]
    except Exception as error:  # an error is an observation
        OBSERVED[name] = ["error", type(error).__name__, clean(str(error))]


_extract = kernel_impl.TaichiCallableTemplateMapper.extract


def counting_extract(self, args):
    COUNTS[id(self)] = COUNTS.get(id(self), 0) + 1
    return _extract(self, args)


kernel_impl.TaichiCallableTemplateMapper.extract = counting_extract


def section(label, name, calls):
    # calls: zero-argument functions, one kernel call each; records their results and the kernel's key extractions
    mapper = MAPPERS[name]
    before = COUNTS.get(id(mapper), 0)
    results = []
    for call in calls:
        try:
            results.append(clean(repr(call())))
        except Exception as error:  # an error is an observation
            results.append(type(error).__name__ + ": " + clean(str(error)))
    OBSERVED["section " + label] = results
    EXTRACTED[label] = [len(calls), COUNTS.get(id(mapper), 0) - before]


def key_shapes(name):
    out = []
    for key, instance in sorted(MAPPERS[name].mapping.items(), key=lambda item: item[1]):
        shape = []
        for entry in key:
            if isinstance(entry, weakref.ref):
                shape.append("weak " + ("alive" if entry() is not None else "dead"))
            elif isinstance(entry, str):
                shape.append(entry)
            else:
                shape.append(type(entry).__name__)
        out.append([instance, shape])
    return out


x = ti.field(ti.f32, shape=4)
y = ti.field(ti.f32, shape=4)
x.from_numpy(np.arange(4, dtype=np.float32))


@ti.kernel
def scale(src: ti.template(), dst: ti.template(), factor: ti.f32):
    for i in src:
        dst[i] = src[i] * factor


@ti.kernel
def pair_sum(fields: ti.template()) -> ti.f32:
    s = 0.0
    for j in range(4):
        for k in ti.static(range(2)):
            s += fields[k][j]
    return s


@ti.kernel
def array_sum(arr: ti.types.ndarray()) -> ti.f32:
    s = 0.0
    for I in ti.grouped(arr):
        s += arr[I]
    return s


@ti.kernel
def echo(n: ti.template()) -> ti.i32:
    return n


@ti.kernel
def maybe(a: ti.template()) -> ti.i32:  # keyed by a data-oriented object (weakly) or by None (as itself)
    return 7


@ti.data_oriented
class Holder:
    def __init__(self, n):
        self.data = ti.field(ti.f32, shape=n)

    @ti.kernel
    def fill(self, value: ti.f32):
        for i in self.data:
            self.data[i] = value + i

    @ti.kernel
    def copy_from(self, other: ti.template()):
        for i in self.data:
            self.data[i] = other.data[i] * 2


for name, kernel in dict(scale=scale, pair_sum=pair_sum, array_sum=array_sum, echo=echo, maybe=maybe,
                         fill=Holder.fill, copy_from=Holder.copy_from).items():
    MAPPERS[name] = kernel._primal.mapper

scale(y, x, 1.0)
section("same template objects, other scalars", "scale",
        [lambda f=f: (scale(x, y, f), y.to_numpy().tolist())[1] for f in (1.0, 2.0, 3.0)])
section("alternating template objects", "scale",
        [lambda a=a, b=b: (scale(a, b, 1.5), b.to_numpy().tolist())[1] for a, b in ((y, x), (x, y), (y, x), (x, y))])
section("keywords after a positional call", "scale",
        [lambda: (scale(src=x, dst=y, factor=2.5), y.to_numpy().tolist())[1]] * 2)
section("a wrong argument count after a positional call", "scale", [lambda: scale(x, y)] * 2)
pair = (x, y)
section("a tuple, the same object", "pair_sum", [lambda: pair_sum(pair)] * 2)
section("a tuple, a new one per call", "pair_sum", [lambda: pair_sum((x, y))] * 2)
array = np.arange(4, dtype=np.float32)
section("an array", "array_sum", [lambda: array_sum(array)] * 2)
array.shape = (2, 2)
section("the same array, reshaped", "array_sum", [lambda: array_sum(array)] * 2)
section("a small int, the same object", "echo", [lambda: echo(3)] * 2)
section("a large int, a new object per call", "echo", [lambda: echo(int("1000001"))] * 2)
section("a computed small int", "echo", [lambda: echo(len("abc"))] * 2)

h1, h2 = Holder(3), Holder(3)
section("a data-oriented object", "fill", [lambda v=v: (h1.fill(v), h1.data.to_numpy().tolist())[1] for v in (1, 2, 3)])
section("another data-oriented object", "fill", [lambda: (h2.fill(5.0), h2.data.to_numpy().tolist())[1]])
h3 = Holder(3)
h3.fill(7.0)
section("a data-oriented argument", "copy_from", [lambda: (h1.copy_from(h3), h1.data.to_numpy().tolist())[1]] * 2)
dead = weakref.ref(h3)
del h3
gc.collect()
see("a data-oriented argument after del", lambda: dead() is None)
h4 = Holder(3)
h4.fill(9.0)
section("the next data-oriented argument", "copy_from",
        [lambda: (h1.copy_from(h4), h1.data.to_numpy().tolist())[1]] * 2)
h5 = Holder(5)
section("a data-oriented argument of a kernel that also takes None", "maybe", [lambda: maybe(h5)] * 2)
dead5 = weakref.ref(h5)
del h5
gc.collect()
see("that argument after del", lambda: dead5() is None)
section("None after a dead weakly keyed argument", "maybe", [lambda: maybe(None)] * 2)
see("the last caller of fill", lambda: (h2.fill(6.0), h2.data.to_numpy().tolist())[1])
gone = weakref.ref(h2)
del h2
gc.collect()
see("a caller after del", lambda: gone() is None)
h6, h7 = Holder(3), Holder(3)
h6.fill(1.0)
h7.fill(2.0)
section("data-oriented objects in turn", "fill",
        [lambda h=h: (h.fill(3.0), h.data.to_numpy().tolist())[1] for h in (h6, h7, h6, h7, h7)])
h8 = Holder(3)
h8.fill(4.0)


def replace_the_field():
    h8.data = ti.field(ti.f32, shape=3)
    h8.data.fill(100.0)


section("a data-oriented argument, its field replaced between the calls", "copy_from",
        [lambda: (h1.copy_from(h8), h1.data.to_numpy().tolist())[1],
         lambda: (replace_the_field(), h1.copy_from(h8), h1.data.to_numpy().tolist())[2]])
section("True, the int 1, True", "echo", [lambda: echo(True), lambda: echo(1), lambda: echo(True)])
section("the int 1, the float 1.0, the int 1", "maybe", [lambda: maybe(1), lambda: maybe(1.0), lambda: maybe(1)])
see("the last echo before ti.init", lambda: echo(3))
for name in MAPPERS:
    see("instances " + name, lambda name=name: sorted(MAPPERS[name].mapping.values()))
    see("keys " + name, lambda name=name: key_shapes(name))

ti.reset()
ti.init(CPU_ARGUMENTS)
z = ti.field(ti.f32, shape=4)
w = ti.field(ti.f32, shape=4)
z.from_numpy(np.ones(4, dtype=np.float32))
section("after ti.init, new fields", "scale", [lambda: (scale(z, w, 2.0), w.to_numpy().tolist())[1]] * 2)
see("instances scale after ti.init", lambda: sorted(MAPPERS["scale"].mapping.values()))
see("keys scale after ti.init", lambda: key_shapes("scale"))
section("after ti.init, the key arguments of the last call", "echo", [lambda: echo(3)] * 2)
see("instances echo after ti.init", lambda: sorted(MAPPERS["echo"].mapping.values()))
h9 = Holder(3)
section("after ti.init, a new data-oriented object", "fill",
        [lambda v=v: (h9.fill(v), h9.data.to_numpy().tolist())[1] for v in (1.0, 2.0)])
see("instances fill after ti.init", lambda: sorted(MAPPERS["fill"].mapping.values()))
print("OBSERVED " + json.dumps(OBSERVED, sort_keys=True))
print("EXTRACTED " + json.dumps(EXTRACTED, sort_keys=True))
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
    """The scenarios with fast path f on (TI_LAUNCH_FASTPATH unset) and off (0, Taichi 1.7.4's code)."""
    return (
        run_script(tmp_path_factory.mktemp("on"), SCENARIOS, "unset"),
        run_script(tmp_path_factory.mktemp("off"), SCENARIOS, "0"),
    )


def test_every_result_key_and_instance_is_taichi_1_7_4s(runs):
    """Every observation is equal on and off: kernel results while template arguments repeat, alternate, come with
    keywords or a wrong count, are tuples (the same object or a new one), arrays (reshaped in place), small and large
    ints and data-oriented objects; a data-oriented argument and a caller that die (no reused key keeps them alive)
    and the objects after them, None after a dead weakly keyed argument; every kernel's instances and key shapes (weak
    references alive or dead), also after ti.reset and ti.init, where a kernel whose key arguments repeat compiles
    again; data-oriented objects in turn; a data-oriented argument whose field is replaced (the kernel instance keeps
    the field it compiled with, on and off); equal bool and int, and int and float, template arguments share an
    instance."""
    on, off = (tagged(run, "OBSERVED") for run in runs)
    assert set(on) == set(off)
    differing = {name: (on[name], off[name]) for name in on if on[name] != off[name]}
    assert not differing, differing
    assert on["a data-oriented argument after del"] == ["value", "True"]
    assert on["a caller after del"] == ["value", "True"]
    assert on["that argument after del"] == ["value", "True"]
    # The dead argument's weak reference gives None, and None is a template argument of its own: 1.7.4 keys it
    # afresh, a second kernel instance.
    assert on["section None after a dead weakly keyed argument"] == ["7", "7"]
    assert on["instances maybe"] == ["value", "[0, 1, 2]"]
    assert on["keys maybe"] == ["value", "[[0, ['weak dead']], [1, ['NoneType']], [2, ['int']]]"]
    # the kernel keeps the field it compiled with, on and off, so the second copy reads 4.0 + i, not 100
    replaced = on["section a data-oriented argument, its field replaced between the calls"]
    assert replaced[0] == replaced[1] == "[8.0, 10.0, 12.0]"
    assert on["section True, the int 1, True"] == ["1", "1", "1"]
    assert on["section after ti.init, the key arguments of the last call"] == ["3", "3"]


# Key extractions on, per section; off, every call extracts its key.
EXTRACTIONS_ON = {
    "same template objects, other scalars": 1,  # the first call's src and dst differ from the last call's
    "alternating template objects": 4,
    "keywords after a positional call": 0,  # keywords reach lookup through _process_args: the same objects
    "a wrong argument count after a positional call": 0,  # _process_args refuses before any lookup
    "a tuple, the same object": 2,  # Taichi keys a tuple per instance
    "a tuple, a new one per call": 2,
    "an array": 2,  # an ndarray parameter is keyed by the array's state
    "the same array, reshaped": 2,
    "a small int, the same object": 1,
    "a large int, a new object per call": 2,
    "a computed small int": 1,  # len("abc") is the cached int 3: the first call follows a large int, the second reuses
    "a data-oriented object": 1,
    "another data-oriented object": 1,
    "a data-oriented argument": 1,
    "the next data-oriented argument": 1,  # the weak reference to the dead argument gives None, so a fresh key
    "a data-oriented argument of a kernel that also takes None": 1,
    "None after a dead weakly keyed argument": 1,  # the dead weak reference never matches, not even None
    "data-oriented objects in turn": 4,  # the fifth call repeats the fourth's object
    "a data-oriented argument, its field replaced between the calls": 1,  # the key never reads a field
    "True, the int 1, True": 3,  # equal keys, other objects: the same instance through the full key
    "the int 1, the float 1.0, the int 1": 3,
    "after ti.init, new fields": 1,
    "after ti.init, the key arguments of the last call": 0,  # the same key, and the kernel compiles again
    "after ti.init, a new data-oriented object": 1,
}


def test_fast_path_f_saves_the_key_of_a_repeated_call_only(runs):
    """Off, every call that reaches lookup extracts its template key; on, a call whose key arguments are the objects
    of the kernel's last call extracts none, and every other call (other objects, a tuple, an array parameter, the
    object after a dead one, the first call after ti.init) extracts its key."""
    on, off = (tagged(run, "EXTRACTED") for run in runs)
    assert set(on) == set(off) == set(EXTRACTIONS_ON)
    refused = "a wrong argument count after a positional call"
    assert {label: extracted for label, (calls, extracted) in off.items() if extracted != calls} == {refused: 0}
    assert {label: extracted for label, (calls, extracted) in on.items()} == EXTRACTIONS_ON
