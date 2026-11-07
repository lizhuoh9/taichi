# Added in the fork lizhuoh9/taichi on 2026-11-06 (see FORK.md): tests of the launch fast paths and of their switch.
"""The launch fast paths of taichi.lang.kernel_impl (FORK.md) return what Taichi 1.7.4's code returns and raise what
it raises.  The risks: a template key other than Taichi's (another kernel instance, or a key that keeps an object
alive), a scalar set into the wrong slot or with another conversion, a lost type error or C++ error conversion, a
compiled kernel's data kept past its program or launched after a compile-config write, a call that should go to the
released code and does not, and a switch that is not what the process runs.

Each check runs a script in a fresh interpreter on the CPU (Taichi reads a kernel's source from its file, and the
fast paths keep per-type and per-kernel state), with the switch flipped through kernel_impl._LAUNCH_FASTPATH between
two phases of one process, or through TI_LAUNCH_FASTPATH between two processes.
"""

import json
import os
import subprocess
import sys

import pytest

import taichi as ti
from taichi import _fork
from taichi.lang import kernel_impl

CPU = "arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False"


def run_script(directory, text, *args, switch=None):
    """Run `text` as a script file in a fresh interpreter; `switch` sets TI_LAUNCH_FASTPATH ("unset" removes it).
    Returns the finished process, which must exit 0."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "check.py"
    script.write_text(text, encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", TI_SKIP_VERSION_CHECK="ON")
    if switch == "unset":
        env.pop("TI_LAUNCH_FASTPATH", None)
    elif switch is not None:
        env["TI_LAUNCH_FASTPATH"] = switch
    done = subprocess.run(
        [sys.executable, "-B", str(script), *map(str, args)],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return done


def tagged(done, tag):
    """The JSON payload of the one stdout line that starts with `tag`."""
    (line,) = [line for line in done.stdout.splitlines() if line.startswith(tag + " ")]
    return json.loads(line[len(tag) + 1 :])


# --------------------------------------------------------------------------------------------- the marker and switch
def test_the_marker_names_the_fork_and_its_switch():
    """taichi._fork is what a program that needs this fork reads: its name, its version (the core's release with the
    fork's local label, so the wheel and the core agree) and the switch this process runs with."""
    assert ti._fork is _fork
    assert (_fork.NAME, _fork.VERSION, _fork.BASE) == ("lizhuoh9/taichi", "1.7.4+1", "v1.7.4")
    assert _fork.VERSION.split("+")[0] == ".".join(str(part) for part in ti.__version__)
    assert isinstance(_fork.LAUNCH_FASTPATH, bool) and kernel_impl._LAUNCH_FASTPATH is _fork.LAUNCH_FASTPATH


@pytest.mark.parametrize("value, expected", [(None, True), ("1", True), ("0", False)])
def test_the_switch_reads_unset_one_and_zero(value, expected):
    """Unset or 1 turns the fast paths on, 0 turns them off."""
    environ = {} if value is None else {"TI_LAUNCH_FASTPATH": value}
    assert _fork._read_launch_fastpath(environ) is expected


@pytest.mark.parametrize("value", ["", "2", "true", "off", " 0", "0 ", "01"])
def test_any_other_switch_value_is_refused_by_name(value):
    """Any other value raises ValueError naming the variable and the value: a typo never runs the other way."""
    with pytest.raises(ValueError, match=r"TI_LAUNCH_FASTPATH must be 1 .* or 0 \(off\); got " + repr(value)):
        _fork._read_launch_fastpath({"TI_LAUNCH_FASTPATH": value})


SWITCH = """
import json
import os
import taichi as ti
from taichi.lang import kernel_impl
os.environ["TI_LAUNCH_FASTPATH"] = "1" if ti._fork.LAUNCH_FASTPATH is False else "0"   # read once: no effect now
print("SWITCH", json.dumps([ti._fork.LAUNCH_FASTPATH, kernel_impl._LAUNCH_FASTPATH]))
"""


@pytest.mark.parametrize("switch, expected", [("unset", True), ("1", True), ("0", False)])
def test_the_switch_is_read_once_at_import(tmp_path, switch, expected):
    """The process runs with the value TI_LAUNCH_FASTPATH had at import, in the marker and in kernel_impl alike."""
    assert tagged(run_script(tmp_path, SWITCH, switch=switch), "SWITCH") == [expected, expected]


def test_a_bad_switch_value_stops_the_import(tmp_path):
    """A bad value fails `import taichi` itself, before any program exists, with the variable and the value named."""
    script = tmp_path / "check.py"
    script.write_text("import taichi\n", encoding="utf-8")
    env = dict(os.environ, TI_LAUNCH_FASTPATH="yes", TI_SKIP_VERSION_CHECK="ON", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(
        [sys.executable, "-B", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert done.returncode != 0
    assert "ValueError: TI_LAUNCH_FASTPATH must be 1" in done.stderr and "got 'yes'" in done.stderr


# --------------------------------------------------------------------------------------------------- template keys
KEYS = f"""
import enum
import json
import weakref

import numpy as np
import taichi as ti
from taichi.lang import kernel_impl
from taichi.types import template

ti.init({CPU})
scalar = ti.field(ti.f32, shape=4)
vector = ti.Vector.field(3, ti.f64, shape=4)
matrix = ti.Matrix.field(2, 2, ti.i32, shape=4)


@ti.data_oriented
class Holder:
    def __init__(self):
        self.values = ti.field(ti.f32, shape=2)


class Derived(Holder):
    pass


class Items(list):
    pass


class Mode(enum.IntEnum):
    A = 1


class Meta(type):                   # an attribute of the metaclass: the class sees it, its instances do not
    _data_oriented = True


class WithMeta(metaclass=Meta):
    pass


NAMED = [("_data_oriented", "f4")]  # a structured field: an attribute of an np.record, not of an np.void
ARGS = dict(scalar_field=scalar, vector_field=vector, matrix_field=matrix, python_int=7, python_float=0.5,
            python_bool=True, python_str="mode", none=None, numpy_f32=np.float32(0.25), numpy_i64=np.int64(-3),
            numpy_bool=np.bool_(True), numpy_void=np.zeros(1, dtype=NAMED)[0],
            numpy_record=np.rec.array([(1.0,)], dtype=NAMED)[0],
            numpy_record_plain=np.rec.array([(1.0,)], dtype=[("a", "f4")])[0], data_oriented=Holder(),
            data_oriented_subclass=Derived(), tuple_of_arguments=(scalar, vector, 3, Holder()),
            list_subclass=Items([scalar]), plain_list=[scalar], plain_dict={{"a": scalar}}, snode=scalar.snode,
            expr=scalar.vars[0], core_expr=scalar.vars[0].ptr, ndarray=ti.ndarray(ti.f32, shape=4),
            numpy_array=np.zeros(3), enum_member=Mode.A, metaclass_attribute=WithMeta(),
            struct_field=ti.Struct.field({{"a": ti.f32, "b": ti.i32}}, shape=4),
            vector_value=ti.Vector([1.0, 2.0]), struct_value=ti.Struct(a=1, b=2.0))
CASES = [(name, arg, template()) for name, arg in ARGS.items()]
CASES += [("f32_annotation", 1.0, ti.f32), ("ndarray_annotation", np.zeros(3, np.float32), ti.types.ndarray())]


def outcome(arg, anno):
    try:
        return ("key", kernel_impl.TaichiCallableTemplateMapper.extract_arg(arg, anno, "argument"))
    except Exception as error:
        return ("error", (type(error), str(error)))


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, weakref.ref):
        return a() is not None and a() is b()
    return a is b or (not isinstance(a, np.ndarray) and a == b)


def kind(result):
    if result[0] == "error":
        return result[1][0].__name__
    key = result[1]
    if isinstance(key, weakref.ref):
        return "weak_reference"
    return "argument" if any(key is arg for _, arg, _ in CASES) else type(key).__name__


kernel_impl._LAUNCH_FASTPATH = False
released = [outcome(arg, anno) for _, arg, anno in CASES]
fresh = len(kernel_impl._TEMPLATE_KEY_KINDS) == 0     # the released code decides nothing per type
kernel_impl._LAUNCH_FASTPATH = True
rows = {{}}
for _ in range(2):                  # the first call decides each type, the second uses the decision
    fast = [outcome(arg, anno) for _, arg, anno in CASES]
    for (name, arg, anno), r, f in zip(CASES, released, fast):
        rows.setdefault(name, []).append(same(r, f))
decisions = {{name: kernel_impl._TEMPLATE_KEY_KINDS.get(type(arg)) for name, arg, _ in CASES[:len(ARGS)]}}
kinds = {{name: kind(r) for (name, _, _), r in zip(CASES, released)}}
print("KEYS", json.dumps(dict(fresh=fresh, same=rows, kinds=kinds, decisions=decisions)))
"""
EXPECTED_KEYS = {  # the released code's answer: the argument itself, a weak reference, another key, an error
    "scalar_field": "argument",
    "vector_field": "argument",
    "matrix_field": "argument",
    "python_int": "argument",
    "python_float": "argument",
    "python_bool": "argument",
    "python_str": "argument",
    "none": "argument",
    "numpy_f32": "argument",
    "numpy_i64": "argument",
    "numpy_bool": "argument",
    "numpy_void": "argument",
    "numpy_record": "weak_reference",
    "numpy_record_plain": "argument",
    "data_oriented": "weak_reference",
    "data_oriented_subclass": "weak_reference",
    "tuple_of_arguments": "tuple",
    "list_subclass": "weak_reference",
    "plain_list": "TypeError",
    "plain_dict": "TypeError",
    "snode": "SNode",
    "expr": "int",
    "core_expr": "int",
    "ndarray": "TaichiRuntimeTypeError",
    "numpy_array": "argument",
    "enum_member": "argument",
    "metaclass_attribute": "argument",
    "struct_field": "argument",
    "vector_value": "argument",
    "struct_value": "argument",
    "f32_annotation": "str",
    "ndarray_annotation": "tuple",
}
FAST_BRANCH = {"argument": 1, "weak_reference": 2}  # kernel_impl._KEY_IS_ARGUMENT, _KEY_IS_WEAK_REFERENCE
NOT_LISTED = (  # the released branches decide these per instance
    "numpy_void",
    "numpy_record",
    "numpy_record_plain",
    "list_subclass",
    "numpy_array",
    "enum_member",
    "metaclass_attribute",
    "vector_value",
    "struct_value",
)


def test_every_template_key_equals_the_released_one(tmp_path):
    """Each argument kind's key (or error) with the fast path equals the released code's, on its first call (which
    decides its type) and on its second; the fast branch is taken for the listed types only (scalar, vector, matrix
    and struct fields among them).  Not listed: a NumPy array, an enum member, a Python-scope vector or struct value,
    an instance of a class whose metaclass has _data_oriented (the class finds it, the instance does not), and NumPy's
    structured scalars, whose fields belong to each instance (two np.record instances get different keys, so no
    per-type decision fits them).  Non-template annotations never reach the fast branch."""
    keys = tagged(run_script(tmp_path, KEYS), "KEYS")
    assert keys["fresh"] is True
    assert keys["kinds"] == EXPECTED_KEYS
    assert keys["same"] == {name: [True, True] for name in EXPECTED_KEYS}
    listed = {name for name, kind in EXPECTED_KEYS.items() if kind in FAST_BRANCH and name not in NOT_LISTED}
    assert keys["decisions"] == {
        name: FAST_BRANCH[EXPECTED_KEYS[name]] if name in listed else 0 for name in keys["decisions"]
    }


# ------------------------------------------------------------------------------------------------------- launches
LAUNCHES = f"""
import json
import re
import sys

import numpy as np
import taichi as ti
from taichi.lang import kernel_impl

DEBUG, HOW = sys.argv[1] == "debug", sys.argv[2]   # debug: bounds checks; HOW: "toggle" (two phases) or "once"
ti.init({CPU}, debug=DEBUG)
ints, uints = ti.field(ti.i32, shape=5), ti.field(ti.u32, shape=5)
f32s, f64s = ti.field(ti.f32, shape=5), ti.field(ti.f64, shape=5)
behind_i32, behind_f64 = ti.field(ti.i32, shape=2), ti.field(ti.f64, shape=2)
wide, uwide, python_float = ti.field(ti.i64, shape=(2, 5)), ti.field(ti.u64, shape=2), ti.field(ti.f32, shape=2)
vector, matrix = ti.Vector.field(3, ti.f32, shape=()), ti.Matrix.field(2, 2, ti.f32, shape=())
sums = ti.field(ti.i64, shape=2)
pair_a, pair_b, used = ti.field(ti.i32, shape=()), ti.field(ti.f32, shape=()), ti.field(ti.f32, shape=2)
Pair = ti.types.struct(a=ti.i32, b=ti.f32)


@ti.kernel
def put(slot: ti.i32, a: ti.i32, b: ti.u32, c: ti.f32, d: ti.f64):
    ints[slot] = a
    uints[slot] = b
    f32s[slot] = c
    f64s[slot] = d


@ti.kernel
def put_wide(slot: ti.i32, a: ti.i64, b: ti.u64, c: ti.i8, d: ti.u8, e: int, f: float, g: bool):
    wide[slot, 0] = a
    uwide[slot] = b
    wide[slot, 1] = c
    wide[slot, 2] = d
    wide[slot, 3] = e
    python_float[slot] = f
    wide[slot, 4] = g


@ti.kernel
def put_vector(v: ti.types.vector(3, ti.f32)):
    vector[None] = v


@ti.kernel
def put_matrix(m: ti.types.matrix(2, 2, ti.f32)):
    matrix[None] = m


@ti.kernel
def put_pair(p: Pair):
    pair_a[None] = p.a
    pair_b[None] = p.b


@ti.func
def scaled(x, factor=2.0):          # a default and a keyword: _process_args at compile time
    return x * factor


@ti.kernel
def use_func(v: ti.f32):
    used[0] = scaled(v)
    used[1] = scaled(v, factor=3.0)


@ti.pyfunc
def blend(a, b=2.0):                # called from Python scope: _process_args with a default
    return a * b


@ti.kernel
def say(a: ti.i32):
    print("SAY", a)


@@SUM_KERNELS@@


@ti.kernel
def put_behind_templates(first: ti.template(), a: ti.i32, second: ti.template(), d: ti.f64):
    first[1] = a
    second[1] = d


@ti.kernel
def triple(a: ti.f64) -> ti.f64:
    return a * 3.0


@ti.kernel
def negate(a: ti.i32) -> ti.i32:
    return -a


@ti.kernel
def spread(a: ti.f32) -> ti.types.vector(2, ti.f32):
    return ti.Vector([a, 2 * a])


@ti.kernel
def both(a: ti.i32, b: ti.f64) -> tuple[ti.i32, ti.f64]:
    return a + 1, b * 2


@ti.kernel
def scale(values: ti.types.ndarray(), factor: ti.f32):
    for i in values:
        values[i] *= factor


@ti.kernel
def poke(i: ti.i32):
    ints[i] = 1


@ti.data_oriented
class Accumulator:
    def __init__(self):
        self.total = ti.field(ti.f64, shape=())

    @ti.kernel
    def add(self, weight: ti.f64, count: ti.i32):
        self.total[None] += weight * count


@ti.kernel
def bump(holder: ti.template(), by: ti.f64):
    holder.total[None] += by


accumulator = Accumulator()
NAMED = [("a", "i4")]
VALUES = [(0, -2 ** 31, 2 ** 32 - 1, 0.1, 0.1), (1, 2 ** 31 - 1, 0, 3.4028234663852886e38, 1.0e308),
          (2, np.int64(-7), np.uint32(123456789), np.float32(1.0e-45), np.float64(-0.0)), (3, True, False, 7, -2)]
WIDE = [(0, -2 ** 63, 2 ** 64 - 1, -128, 255, -7, 0.1, True), (1, 2 ** 63 - 1, 0, 127, 0, 2 ** 31 - 1, -2.5, False)]
BAD = [("x", 1, 1, 1.0, 1.0), (0, 1.5, 1, 1.0, 1.0), (0, 1, 1, "1.0", 1.0), (0, 1, 1, 1.0, None), (0, 1, -1, 1.0, 1.0),
       (0, 1), (0, 1, 2, 3.0, 4.0, 5), (0, np.zeros(1, dtype=NAMED)[0], 1, 1.0, 1.0),
       (0, 1, np.rec.array([(1,)], dtype=NAMED)[0], 1.0, 1.0)]
BAD_KEYWORDS = [dict(slot=0, a=1, b=2, c=3.0, zz=4.0), dict(a=1, b=2, c=3.0, d=4.0)]


def failure(call):
    try:
        call()
    except Exception as error:      # a pybind message names the launch context's address, new in every call
        return [type(error).__module__ + "." + type(error).__qualname__, re.sub("0x[0-9a-f]+", "0x", str(error))]
    return None


def phase():
    for field in (ints, uints, f32s, f64s, behind_i32, behind_f64, accumulator.total, wide, uwide, python_float, sums,
                  pair_a, pair_b, used):
        field.fill(0)
    for values in VALUES:
        put(*values)
    for values in WIDE:
        put_wide(*values)
    put(slot=4, a=11, b=12, c=13.5, d=14.25)
    put(4, 21, c=23.5, b=22, d=24.25)
    put_behind_templates(behind_i32, 77, behind_f64, 0.3)
    put_vector(ti.Vector([1.5, -2.0, 0.25]))
    put_matrix(ti.Matrix([[1.0, 2.0], [3.0, 4.0]]))
    put_pair(Pair(a=3, b=0.5))
    use_func(1.25)
    accumulator.add(0.1, 3)
    bump(accumulator, 0.5)
    say(5)
    poke(3)                         # ints[3] is 1 already (True above)
    sum64(*range(64))
    array = np.arange(4, dtype=np.float32)
    scale(array, 0.5)
    returned = [triple(0.1), triple(np.float32(0.2)), negate(-2 ** 31 + 1), negate(np.int32(5)), spread(0.75).to_list(),
                list(both(4, 0.25)), blend(3.0), blend(3.0, 4.0), blend(b=5.0, a=1.0)]
    fields = [field.to_numpy().tolist() for field in (ints, uints, f32s, f64s, behind_i32, behind_f64, wide, uwide,
                                                      python_float, vector, matrix, sums, pair_a, pair_b, used)]
    errors = [failure(lambda bad=bad: put(*bad)) for bad in BAD]
    errors += [failure(lambda bad=bad: put(**bad)) for bad in BAD_KEYWORDS]
    errors += [failure(lambda: put(0, 1, 2, 3.0, slot=4.0)), failure(lambda: sum65(*range(65))),
               failure(lambda: blend()), failure(lambda: blend(1.0, 2.0, 3.0)), failure(lambda: put_vector(1.0))]
    if DEBUG:      # last: the bounds error still writes ints[10]
        errors.append(failure(lambda: poke(10)))
    instances = [len(kernel._primal.mapper.mapping) for kernel in (put, put_behind_templates, triple, scale, bump)]
    instances.append(len(Accumulator.add._primal.mapper.mapping))
    return dict(fields=fields, returned=returned, scaled=array.tolist(), total=accumulator.total[None], errors=errors,
                instances=instances)


KERNELS = dict(put=put, put_wide=put_wide, put_vector=put_vector, put_matrix=put_matrix, put_pair=put_pair,
               use_func=use_func, say=say, sum64=sum64, sum65=sum65, put_behind_templates=put_behind_templates,
               triple=triple, negate=negate, spread=spread, both=both, scale=scale, poke=poke, bump=bump,
               add=Accumulator.add)


def plans():
    shapes = {{}}
    for name, kernel in KERNELS.items():
        plan = kernel._primal._scalar_launch_plan
        shapes[name] = "unplanned" if plan is kernel_impl._UNPLANNED else (None if plan is None else len(plan[1]))
    return shapes


if HOW == "toggle":
    kernel_impl._LAUNCH_FASTPATH = False
    released = phase()
    untouched = plans()
    kernel_impl._LAUNCH_FASTPATH = True
    fast = phase()
    print("LAUNCHES", json.dumps(dict(released=released, fast=fast, untouched=untouched, plans=plans())))
else:
    result = phase()
    kept = {{name: kernel._primal._kept_compiled_kernel_data is not None for name, kernel in KERNELS.items()}}
    print("ONCE", json.dumps(dict(result=result, plans=plans(), kept=kept, switch=kernel_impl._LAUNCH_FASTPATH)))
"""
SUM_KERNELS = "\n\n".join(  # 64 scalars take the fast path; one more goes to the released code
    "@ti.kernel\ndef sum%d(%s):\n    sums[%d] = %s\n"
    % (
        count,
        ", ".join("a%d: ti.i32" % i for i in range(count)),
        count - 64,
        " + ".join("a%d" % i for i in range(count)),
    )
    for count in (64, 65)
)
LAUNCH_SCRIPT = LAUNCHES.replace("@@SUM_KERNELS@@", SUM_KERNELS)
PLANNED = {  # scalar parameters of the kernels with a plan; None: the released code launches the kernel
    "put": 5,
    "put_wide": 8,
    "put_vector": None,
    "put_matrix": None,
    "put_pair": None,
    "use_func": 1,
    "say": 1,
    "sum64": 64,
    "sum65": None,
    "put_behind_templates": 2,
    "triple": 1,
    "negate": 1,
    "spread": 1,
    "both": 2,
    "scale": None,
    "poke": 1,
    "bump": 1,
    "add": 2,
}


def check_released_values(result, mode):
    """The values that the released code wrote, returned and raised (the fast path is compared with these)."""
    ints, uints, f32s, f64s, behind_i32, behind_f64, wide, uwide, python_float, vector, matrix, sums = result["fields"][
        :12
    ]
    pair_a, pair_b, used = result["fields"][12:]
    assert ints[:2] == [-(2**31), 2**31 - 1] and uints[:2] == [2**32 - 1, 0]
    assert f32s[0] == 0.10000000149011612 and f32s[1] == 3.4028234663852886e38 and f64s[:2] == [0.1, 1.0e308]
    assert (ints[4], uints[4], f32s[4], f64s[4]) == (21, 22, 23.5, 24.25)
    assert wide == [[-(2**63), -128, 255, -7, 1], [2**63 - 1, 127, 0, 2**31 - 1, 0]] and uwide == [2**64 - 1, 0]
    assert python_float == [0.10000000149011612, -2.5] and vector == [1.5, -2.0, 0.25] and sums == [2016, 0]
    assert matrix == [[1.0, 2.0], [3.0, 4.0]] and (pair_a, pair_b) == (3, 0.5) and used == [2.5, 3.75]
    assert (behind_i32[1], behind_f64[1], result["total"]) == (77, 0.3, 0.1 * 3 + 0.5)
    assert result["returned"] == [0.30000000000000004, 3 * 0.20000000298023224, -(-(2**31) + 1), -5, [0.75, 1.5]] + [
        [5, 0.5],
        6.0,
        12.0,
        5.0,
    ]
    assert result["scaled"] == [0.0, 0.5, 1.0, 1.5] and result["instances"] == [1, 1, 1, 1, 1, 1]
    kinds = [error[0] for error in result["errors"]]
    assert kinds[:4] == ["taichi.lang.exception.TaichiRuntimeTypeError"] * 4  # wrong types
    assert kinds[4] == "builtins.TypeError"  # -1 into an unsigned slot
    assert kinds[5:7] == ["taichi.lang.exception.TaichiSyntaxError"] * 2  # missing, too many
    assert kinds[7:9] == ["taichi.lang.exception.TaichiRuntimeTypeError"] * 2  # np.void, np.record
    assert kinds[9:12] == ["taichi.lang.exception.TaichiSyntaxError"] * 3  # unexpected, missing, multiple
    assert kinds[12] == "taichi.lang.exception.TaichiRuntimeError" and "Do not exceed 64" in result["errors"][12][1]
    assert kinds[13:15] == ["taichi.lang.exception.TaichiSyntaxError"] * 2  # pyfunc: missing, too many
    assert kinds[15] is not None  # a float for a vector
    assert kinds[16:] == (["taichi.lang.exception.TaichiAssertionError"] if mode == "debug" else [])


@pytest.mark.parametrize("mode", ["release", "debug"])
def test_launches_set_values_and_raise_as_the_released_code(tmp_path, mode):
    """The same kernels run once on the released code and then on the fast paths, in one process, without and with
    debug mode: every field, return value and error is equal, the fast path compiles no new kernel instance (its
    template keys hit the released ones), the extreme values of every integer width, of Python's int, float and bool
    and of f32 and f64 reach the kernel exactly, and the printing kernel prints in both phases.  Scalar kernels
    (also behind templates, data-oriented methods and kernels returning values) get a plan; kernels with a vector, a
    matrix, a struct or an ndarray parameter, and one with 65 scalars, are handed to the released code."""
    done = run_script(tmp_path, LAUNCH_SCRIPT, mode, "toggle")
    launches = tagged(done, "LAUNCHES")  # JSON keeps every float's bits
    assert launches["fast"] == launches["released"]
    assert launches["untouched"] == {name: "unplanned" for name in PLANNED}
    assert launches["plans"] == PLANNED
    assert done.stdout.count("SAY 5") == 2
    check_released_values(launches["released"], mode)


def test_the_switch_off_launches_exactly_as_on(tmp_path):
    """Between two processes: TI_LAUNCH_FASTPATH=0 makes no plan and keeps no compiled data, unset makes them, and
    both write, return and raise the same."""
    off = tagged(run_script(tmp_path / "off", LAUNCH_SCRIPT, "release", "once", switch="0"), "ONCE")
    on = tagged(run_script(tmp_path / "on", LAUNCH_SCRIPT, "release", "once", switch="unset"), "ONCE")
    assert (off["switch"], on["switch"]) == (False, True)
    assert off["result"] == on["result"]
    check_released_values(on["result"], "release")
    assert off["plans"] == {name: "unplanned" for name in PLANNED} and on["plans"] == PLANNED
    assert not any(off["kept"].values())
    assert on["kept"] == {name: planned is not None for name, planned in PLANNED.items()}


# ------------------------------------------------------------------------------------------- kept compiled data
LIFETIME = f"""
import gc
import json
import weakref

import taichi as ti
from taichi.lang import impl, kernel_impl

kernel_impl._LAUNCH_FASTPATH = True
x = y = None


@ti.kernel
def fill(value: ti.f32):
    for i in x:
        x[i] = value


@ti.kernel
def fill_y(value: ti.f32):
    for i in y:
        y[i] = value


def tree_of_y():
    global y
    builder = ti.FieldsBuilder()
    y = ti.field(ti.f32)
    builder.dense(ti.i, 3).place(y)
    return builder.finalize()


report = {{}}
for renewal in ("init", "reset"):
    ti.init({CPU})
    x = ti.field(ti.f32, shape=3)
    fill(1.5)
    prog, kept = fill._primal._kept_compiled_kernel_data
    handles = [weakref.ref(data) for data in kept.values()]
    before = dict(kept=len(kept), program=prog is impl.get_runtime().prog, values=x.to_numpy().tolist())
    del prog, kept
    if renewal == "reset":
        ti.reset()
    ti.init({CPU})
    gc.collect()
    dropped = all(handle() is None for handle in handles)
    fresh = fill._primal._kept_compiled_kernel_data is None
    x = ti.field(ti.f32, shape=3)
    fill(2.5)
    report[renewal] = dict(before=before, dropped=dropped, fresh=fresh, after=x.to_numpy().tolist(),
                           kept=len(fill._primal._kept_compiled_kernel_data[1]))
first_tree = tree_of_y()
fill_y(1.0)
first = y.to_numpy().tolist()
first_tree.destroy()               # SNodeTree.destroy clears every kernel's compiled instances: they compile again
tree_of_y()
fill_y(2.0)
kept_y = len(fill_y._primal._kept_compiled_kernel_data[1])
report["destroy"] = dict(first=first, after=y.to_numpy().tolist(), kept=kept_y)
print("LIFETIME", json.dumps(report))
"""


def test_the_kept_compiled_kernel_data_is_dropped_with_its_program(tmp_path):
    """ti.init and ti.reset drop every kernel's kept data (the old handles are freed: no stale handle, no leak), and
    the same kernel compiles and runs on the new program.  Within one program, SNodeTree.destroy makes the kernels
    compile again: the new compiled kernel gets its own data, and the kernel writes the new tree's field."""
    report = tagged(run_script(tmp_path, LIFETIME), "LIFETIME")
    for renewal in ("init", "reset"):
        assert report[renewal] == dict(
            before=dict(kept=1, program=True, values=[1.5] * 3), dropped=True, fresh=True, after=[2.5] * 3, kept=1
        )
    assert report["destroy"] == dict(first=[1.0] * 3, after=[2.0] * 3, kept=2)


CONFIG_WRITE = f"""
import json

import taichi as ti
from taichi.lang import impl, kernel_impl

kernel_impl._LAUNCH_FASTPATH = True
ti.init({CPU}, opt_level=0)
x = ti.field(ti.f32, shape=4, needs_grad=True)
loss = ti.field(ti.f32, shape=(), needs_grad=True)


@ti.kernel
def fill(value: ti.f32):
    for i in x:
        x[i] = value


@ti.kernel
def total():
    for i in x:
        loss[None] += x[i] ** 2


def launched_as_taichi(kernel):
    # the kept data of the kernel's compiled instance is what launch_kernel's general path asks for now
    prog, kept = kernel._kept_compiled_kernel_data
    (t_kernel,) = kernel.compiled_kernels.values()
    return prog is impl.get_runtime().prog and kept[t_kernel] is prog.compile_kernel(
        prog.config(), prog.get_device_caps(), t_kernel)


prog = impl.get_runtime().prog
fill(1.5)
total()
before = [prog.config().opt_level, launched_as_taichi(fill._primal), launched_as_taichi(total._primal)]
loss.grad[None] = 1.0
total.grad()                        # autodiff at opt_level 0: Kernel.__call__ writes opt_level = 1
fill(2.5)
total()
after = [prog.config().opt_level] + [launched_as_taichi(kernel) for kernel in (fill._primal, total._primal, total.grad)]
print("CONFIG", json.dumps(dict(before=before, after=after, values=x.to_numpy().tolist(),
                                grad=x.grad.to_numpy().tolist(), loss=loss[None])))
"""


def test_a_compile_config_write_keeps_the_data_taichi_launches(tmp_path):
    """The kept data is keyed by program and compiled kernel, not by the compile config: after an autodiff call at
    opt_level 0 (Kernel.__call__ writes opt_level = 1), the kept data of every kernel, compiled before or after the
    write, is what the general path asks prog.compile_kernel for, and the values and the gradient are right.  Should
    a Taichi recompile a compiled kernel after such a write, this fails: the kept data would need the config in its
    key."""
    report = tagged(run_script(tmp_path, CONFIG_WRITE), "CONFIG")
    assert report == dict(
        before=[0, True, True], after=[1, True, True, True], values=[2.5] * 4, grad=[3.0] * 4, loss=34.0
    )
