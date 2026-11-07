# The fork lizhuoh9/taichi

This repository is an unofficial fork of [taichi-dev/taichi](https://github.com/taichi-dev/taichi), not affiliated with the
Taichi project. Taichi is licensed under the Apache License 2.0 (`LICENSE`, unchanged), and so are the fork's changes.

## Base

- Branch `main`, built on the upstream tag `v1.7.4` (commit `b4b956fdaa67fd3e198527fd607053ecb9bf4aa2`, the commit the official
  1.7.4 wheel reports). The branch `master` stays upstream's.
- The fork changes Python files only. The C++ core is the release's: the fork's wheel carries the official wheel's binaries byte
  for byte, so `ti.__version__` stays `(1, 7, 4)`. The fork's own version is `taichi._fork.VERSION`.

## Versions

A version is Taichi's release, then a local label that counts the fork's rounds of changes: `1.7.4+4` is the fourth round on Taichi
1.7.4. A local label satisfies `taichi==1.7.4` (PEP 440), and no public index carries one, so it never collides with an official
release. The first two rounds were released under other labels, and `1.7.4+3` was never released on its own.

| Version | What changed | Measured |
|---|---|---|
| `1.7.4+1` | Less Python work per kernel call: a template argument's key is decided once per type, positional arguments pass straight through, and a kernel with only scalar parameters launches with a plan made at its first launch. Adds the marker `taichi._fork`, the switch `TI_LAUNCH_FASTPATH` and the wheel tool (changes 1 to 5). | [A], official 1.7.4 to `+1`, host microseconds per call: 9.38 to 5.58, 16.32 to 6.17, 51.64 to 21.31 on the CPU backend ([M1]); 24.84 to 18.48, 33.97 to 19.25, 83.88 to 50.34 on CUDA ([M2]). A quasi-2-D fluid-structure case on a 4 x 144 x 864 grid (2,034 kernel launches in a step at 48 solver iterations; the timed steps ran 16 or 32), with the same fast paths as a patch on the official wheel, two pairs of runs each: 0.66 of the step time at the same mix of solver iterations ([M3], the host bounds the step: GPU 37 % busy, then 55 %); 0.977 ([M2], the GPU bounds the step: 89 % busy, then 90 %). |
| `1.7.4+2` | A bug fix: an ndarray of an earlier program, freed by Python after the next `ti.init`, could free a live ndarray of the current program (change 6). | No speed change is claimed. On the official wheel the test's scenario freed 0 to 4 of 256 live ndarrays per run, over 5 runs ([M1]). |
| `1.7.4+3` | A data-oriented class gets a descriptor for each kernel instead of a `__getattribute__` hook, so reading a field or any other plain attribute of an instance is Python's own lookup (change 7). Attribute writes on an instance cost more. | `+2` to an earlier draft of `+3`, microseconds per operation, beside a CPU test suite ([M1]): field read 0.344 to 0.010, kernel read 0.946 to 0.503, attribute write 0.022 to 0.402. Kernel read on a class with 26 classes in its MRO, weighted by a real step's launches, `+2` to `+3`: 1.126 to 0.643, on one thread of the [M1] host with no Taichi backend started, beside a GPU job. |
| `1.7.4+4` | A leaner `Kernel.__call__` and launch, and the template key of a kernel's last call reused while its arguments repeat (changes 8 and 9). | An earlier draft of `+3` to that draft with changes 8 and 9, host microseconds per call, beside a test suite ([M1]): [A] 5.95 to 4.93, 6.75 to 5.07, 22.68 to 13.09. |

[A] is one kernel without arguments, a data-oriented method without arguments, and a data-oriented method with 20 field and 8 scalar
arguments, in that order: the host time per call, medians over rotations of the arms.

- [M1]: one thread of an AMD Ryzen 9 5950X, CPU backend, Linux 6.6 under WSL2, CPython 3.10.12.
- [M2]: the same host with an NVIDIA GeForce 1660 (6 GB) on CUDA.
- [M3]: an AMD EPYC 7352 host with an NVIDIA RTX 3090 on CUDA.

## Changes

Every code file the fork changes or adds carries a header comment with the date and a one-line summary (Apache-2.0,
section 4b).

1. Launch fast paths in `python/taichi/lang/kernel_impl.py`, on by default. Taichi's host time per kernel call is about 9 to 52
   microseconds on the CPU backend (the table above), much of it Python that repeats at every call, and it bounds the step of a
   program that launches many small kernels per step. Each fast path returns what Taichi 1.7.4 returns and raises what it raises, except the
   differences listed under change 7, and hands every case it does not cover to Taichi's code:
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
   (`"lizhuoh9/taichi"`), `VERSION` (`"1.7.4+4"`), `BASE` (`"v1.7.4"`) and `LAUNCH_FASTPATH`. A program that
   needs the fork reads it; stock Taichi has no `taichi._fork`. The underscore keeps Taichi's public namespace as
   released.
4. Tests: `tests/python/test_launch_fastpath.py` compares every argument kind, value, return value and error with the
   fast paths on and off, the kept data's lifetime across `ti.init`, `ti.reset` and `SNodeTree.destroy`, the
   compile-config write and the switch.
5. `misc/fork_repack_wheel.py` builds the fork's wheel (below).
6. `ScalarNdarray.__del__` (`python/taichi/lang/_ndarray.py`) deletes its ndarray only on the program that created
   it, named by a weak reference to that program's runtime (`ti.init` and `ti.reset` replace the runtime). Taichi
   1.7.4 passed the pointer to the current program's `delete_ndarray`: an ndarray of an earlier program that Python
   freed after the next `ti.init` then erased whichever live ndarray of the current program had reused its address,
   and the next kernel that took it failed with "invalid DeviceAllocation". The tests in
   `tests/python/test_ndarray_program_delete.py` check the delete calls, the runtime's lifetime and the failure
   itself. `ArgPack.__del__` and `SparseMatrixBuilder.__del__` follow the same pattern as 1.7.4's and are unchanged.
7. Fast path d, `data_oriented` (`python/taichi/lang/kernel_impl.py`), under the switch of change 2. Taichi 1.7.4
   replaces a data-oriented class's `__getattribute__` with a hook: every attribute read of an instance runs Python
   code, and every kernel read builds a new `_BoundedDifferentiableMethod`. With the fast path, the class gets no
   hook. It gets a descriptor for each name whose lookup finds a kernel (its own or inherited from any base, a
   staticmethod or classmethod kernel included), a property of an undecorated class, a slot, a getset or another
   descriptor whose value the hook tested, and a `__setattr__` that gives a name such a descriptor when an instance
   stores a kernel (or another value that answers to `_is_wrapped_kernel`) under it; dunder names get none, and a field
   or any other plain attribute is read by Python's own lookup. A kernel read gives what the hook gave; an instance
   attribute is read first, as before; an entry replaced on the class in which the lookup found it, or set later on a
   class between that class and the descriptor's, is read in its new form, with the `_is_staticmethod` that the hook
   took from the hierarchy's first decorated class; a subclass gets
   its own descriptors when it is made, and so does a subclass made before its base was decorated. A descriptor stands
   for the class attribute that the lookup found when the descriptor was made: `callable()` answers as for that entry,
   a call calls it, `inspect.signature` and `inspect.unwrap` reach it (`__wrapped__`), attribute reads that the
   descriptor does not answer itself go to it (`__name__`, `__qualname__`, `__module__`, `__doc__`, `__globals__`,
   `__code__`, `_primal`, `_is_wrapped_kernel` and any other), and an instance set of the descriptor stores that entry.
   Where the descriptors would read differently, the class's decorated hierarchy gets the 1.7.4 hook back, and fast
   path d takes its descriptors, `__setattr__` and `__init_subclass__` hooks out of the hierarchy's classes again, so
   that the hierarchy is 1.7.4's: a property whose getter is a kernel, a property of a decorated class (a subclass's
   too) without a getter or defined again earlier in the MRO (the hook read the decorated class's own property
   first), another `__getattribute__` or `__setattr__`, no instance `__dict__`, two decorated hierarchies in one
   MRO, or a class decorated after a decorated subclass (its hook runs under the subclass's and binds each kernel
   with its own flag). The differences from 1.7.4, for a hierarchy on the fast path,
   which `TI_LAUNCH_FASTPATH=0` removes:
   - The class `__dict__`. `vars(Class)` holds a descriptor in place of each of the class's own entries that the hook
     bound or tested, and also one for each such name inherited from a base and for each name an instance stored a
     kernel under; the first decorated class of a hierarchy also holds fast path d's `__setattr__` and
     `__init_subclass__`, and a subclass's own `__init_subclass__` is wrapped. Fast path d sets these entries with
     `type.__setattr__`, so a metaclass's `__setattr__` does not see them, where it saw the hook. `dir(Class)` also
     lists the names an instance stored a kernel under; `inspect.getattr_static(Class, name)` gives the descriptor,
     and so does `inspect.getattr_static(instance, name)`, also where the instance holds a value of that name (the
     descriptor is a data descriptor, so that sets and deletes keep their meaning: `inspect.isdatadescriptor` is
     true); and `inspect.classify_class_attrs` (so `help()`) names the class holding the descriptor as the defining
     class and sees no staticmethod, classmethod or property there. Its `callable()`, calls, signature and attribute
     reads are those of the entry it stands for, except: after that entry is replaced or deleted on a base of the
     class holding the descriptor, or a new entry of the name is set on a class between, the descriptor still stands
     for the old entry, where `getattr_static` gave the new one or raised; and
     for a name an instance stored a kernel under where the class has no entry, `getattr_static(Class, name)` raised
     `AttributeError`, and the descriptor is not callable and answers no attribute of an entry. Where only its type can
     answer, it is not the entry: `type()` names its own class, `inspect.isfunction` and `inspect.isroutine` are false,
     `inspect.getfile` and `inspect.getsource` raise `TypeError`, `inspect.getdoc` does not find a docstring that a
     kernel inherits from a base's kernel of its name, `repr()` and `dir()` are its own, it is neither `is` nor `==`
     the function `Class.name` gives, a subclass holds its own descriptor for an inherited name (so
     `getattr_static(Sub, name) is getattr_static(Base, name)` is false), `copy.deepcopy` raises `TypeError` and
     `pickle.dumps` raises `PicklingError` where 1.7.4 gave the function and its bytes, an attribute set on it raises
     `AttributeError`, except under its own slots' names (`name`, `own`, `raw`, `state`), where the set succeeds and
     goes to the slot (`c.name = "x"` leaves `c.state[0]` as it was), `__wrapped__` gives the entry where the entry's
     own gave the function it wraps (`inspect.unwrap` ends at the same object), and `__get__` gives what an instance's
     read gives, where a function's `__get__` gave a plain bound method.
   - Tools that read the class `__dict__`. `mock.create_autospec(Class)` finds no function there, so it keeps `self` in
     a kernel's signature: `create_autospec(Class).kernel(1)` and `create_autospec(Class)().kernel(1)` raise
     `TypeError`, and `create_autospec(Class)().kernel(1, 2)` passes; 1.7.4: the other way round.
     `mock.patch.object(Class, kernel, autospec=True)` gives a `MagicMock`, where 1.7.4 gave a function.
     `del Sub.inherited` and `monkeypatch.delattr(Sub, inherited)` succeed for a kernel the subclass inherits, by
     deleting its descriptor, and the kernel stays readable; 1.7.4: `AttributeError`.
   - What `vars(Class)` or `inspect.getattr_static` gives, used as a value other than in an instance set on the fast
     path: it stays a descriptor.
     - Set on a class under another name (`Class.alias = vars(Class)[name]`), it keeps its old name: `obj.alias` reads
       what `obj.name` reads, also after the entry of `name` is replaced, and `obj.alias = 3` writes the instance's
       `name`, so that `obj.name` reads 3. 1.7.4: `obj.alias` reads 3 and `obj.name` is untouched, and `obj.alias` stays
       the kernel it was given.
     - Set back on the class it came from, for a name that class inherits (`Mid.k = getattr_static(Mid, "k")`), it is no
       entry, where 1.7.4 had a copy of the kernel on `Mid`: an entry set later on a base of `Mid` is read, where 1.7.4
       read the copy.
     - Set on an unrelated class (`Other.k = vars(Class)[name]`), its read raises `TaichiSyntaxError` ("Please decorate
       class Other with @ti.data_oriented"). 1.7.4: a plain bound method, whose call raised that.
     - In a staticmethod set on a class (`Class.s = staticmethod(vars(Class)[name])`), it reads as itself. 1.7.4: a
       bound kernel.
     - Put into an instance's `__dict__` (`vars(obj)[name] = ...`, `object.__setattr__`), under any name, it reads as
       itself. 1.7.4: a bound kernel.
     - Stored on an instance of a hierarchy on the hook, its read raises `AttributeError` (the hook sets
       `_is_staticmethod` on it). 1.7.4: a bound kernel.
   - A kernel stored on an instance under a name whose class entry needs no descriptor (`None` or another plain value),
     after a second decoration of the class: the decoration takes that name's descriptor out, and the read gives the
     stored value. 1.7.4: a bound kernel.
   - `Class.mro` after an instance stored a kernel under `mro` (a name only the metaclass defines): the descriptor's
     class read does not reach the metaclass. 1.7.4: the metaclass's attribute. The fork: `AttributeError`.
   - An entry set on a class after the class was made, which an instance's lookup then finds before any descriptor of
     its name (it replaced the descriptor of the instance's own class, or its name had none). The fork reads it through
     Python's own lookup; 1.7.4's hook bound a kernel, raised `AssertionError` for another value that answers to
     `_is_wrapped_kernel`, and read a decorated class's own property first, through its getter:
     - a kernel (an alias such as `Base.alias = Base.kernel`, or a kernel set in place of one of the class's own).
       1.7.4: a bound kernel (`_BoundedDifferentiableMethod`). The fork: the plain bound method of the kernel's
       function; calling it runs the kernel for that instance with the same arguments and result, but it has no
       `.grad`. The same after a name the class inherits was set on it and deleted again (`Class.k = f; del Class.k`):
       that took the class's descriptor away.
     - a staticmethod of a kernel (`Base.s = staticmethod(Base.answer)`). 1.7.4: a bound kernel that ran the kernel
       without the instance. The fork: the kernel's function; a call runs the kernel only when its first argument is
       an instance of a hierarchy on the fast path (the call is then 1.7.4's), and otherwise raises as the kernel's
       class stub does: `IndexError` without arguments, `AssertionError` for an instance of a class with the 1.7.4
       hook, `TaichiSyntaxError` ("Please decorate class int with @ti.data_oriented") for any other first argument.
     - a classmethod of a kernel. 1.7.4: a bound kernel that ran the kernel for the instance. The fork: a method bound
       to the class, whose call raises `TaichiSyntaxError` ("Please decorate class type with @ti.data_oriented").
     - another value that answers to `_is_wrapped_kernel` and is no function (a `unittest.mock.Mock`). 1.7.4:
       `AssertionError`. The fork: the value.
     - a property of an undecorated class whose getter returns a kernel. 1.7.4: a bound kernel. The fork: the kernel's
       function. One whose getter is a kernel. 1.7.4: `AssertionError`. The fork: runs the kernel.
     - a property of a decorated class without a getter. 1.7.4: `TypeError` ("'NoneType' object is not callable").
       The fork: `AttributeError`. One whose name a class earlier in the instance's MRO defines. 1.7.4: the property.
       The fork: the earlier entry.
     A staticmethod set on the first decorated class of the hierarchy under the name of a kernel that a class before
     it in the MRO defines, or a staticmethod replaced there, also differs: 1.7.4 bound that kernel with the
     `_is_staticmethod` of the new entry; the fork keeps the flag of the entry as it was when the descriptor was made.
   - `Class.__bases__ = ...` after the class was made. A descriptor keeps standing for the entry its lookup found: a
     name in the old and the new bases reads the old base's kernel, where 1.7.4 read the new one's, and a kernel of a
     new base under a new name reads as a plain bound method, where 1.7.4 bound it.
   - `super().kernel(...)` in a method. 1.7.4: `AssertionError` (`super()` bypassed the hook, and the kernel's stub
     asserted). The fork: runs the kernel for the instance; and `super().name` for a name with a descriptor reads an
     instance attribute of that name first, where `super()` read only the class.
   - `Class.kernel(instance, ...)`, and a call of the descriptor that `vars(Class)` or `inspect.getattr_static` gives
     with an instance first. 1.7.4: `AssertionError`. The fork: runs the kernel for the instance.
   - A kernel (or another value that answers to `_is_wrapped_kernel`) put into an instance's `__dict__` without fast
     path d's `__setattr__` (`vars(instance)[name] = ...`, `object.__setattr__(instance, name, ...)`, or a set before
     the class was decorated), under a name with no descriptor. 1.7.4: read as a bound kernel (or `AssertionError` for a
     value that is no function). The fork: the stored value.
   - A kernel under a dunder name, defined in a class or stored on an instance, gets no descriptor. An explicit read
     (`obj.__name__`). 1.7.4: a bound kernel. The fork: the plain bound method of the class's kernel (which runs the
     kernel when called), or the value stored on the instance. The interpreter's own call of a class's dunder kernel
     (`obj()` for `__call__`), which bypassed the hook. 1.7.4: `AssertionError`. The fork: runs the kernel.
   - A subclass whose `__init_subclass__` chain does not reach fast path d's (a base's `__init_subclass__` that does
     not call `super().__init_subclass__()`) gets its descriptors, and the check that may hand its hierarchy back to
     the hook, at the first attribute set on one of its instances (normally in `__init__`). Until then its instances
     read its own entries as entries set on a class later (the second difference), a property whose getter is a
     kernel included.
   The tests in `tests/python/test_data_oriented_fastpath.py` compare every read, call, error, template key, class
   attribute, signature and attribute read of a class entry with the fast path on and off, check each difference and
   each case handed back to the hook.
8. Fast path e and fast path c inline (`python/taichi/lang/kernel_impl.py`), under the switch of change 2.
   `Kernel.__call__` sends a primal kernel called with one positional argument per parameter (none of them
   `inspect.Parameter.empty`), outside `ti.ad.FwdMode` and `ti.ad.Tape`, straight to `ensure_compiled` and
   `launch_kernel`: for such a call 1.7.4's `_process_args` returns the arguments and its forward-mode, tape and
   `opt_level` branches do nothing. Every other call takes 1.7.4's code: keywords, a missing or an extra argument, a
   gradient or forward-mode kernel, a call inside a tape or forward mode. Fast path c's launch is written inside
   `launch_kernel`, so a launch takes no further Python frame. It returns and raises what 1.7.4 does; the tests in
   `tests/python/test_launch_fastpath_call.py` compare calls and their errors, a tape's record and gradient, forward
   mode and a gradient kernel, with the fast paths on and off.
9. Fast path f, the template key reused (`python/taichi/lang/kernel_impl.py`), under the switch of change 2.
   `TaichiCallableTemplateMapper.lookup` returns its last result while every argument that the key holds, or holds a
   weak reference to, is the same object as at that lookup. It applies to a kernel or function whose non-template
   parameters are all keyed `'#'` (no ndarray, texture, argument pack or sparse matrix builder parameter: their keys
   follow the argument's state) and to template arguments that fast path a keys by the argument itself or by a weak
   reference to it; a tuple or another object that Taichi keys per instance takes the full key. The reused result
   holds no more than 1.7.4's mapping holds: an argument kept in the key is alive, so no other object can have its
   identity, and a weak reference to a dead object never matches (it gives `None`, which is a template argument of
   its own); Taichi never changes an entry of the mapping. It returns and raises what 1.7.4 does; the tests in
   `tests/python/test_launch_fastpath_key.py` count the key extractions and compare every result, key and kernel
   instance with the fast paths on and off, also for `None` after a dead weakly keyed argument.

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
wheels of the other official platforms and Python versions; they are untested, and the descriptors of change 7 lean on CPython's
behaviour.

Install it with `pip install taichi-1.7.4+4-cp310-cp310-manylinux_2_27_x86_64.whl`, into the environment first. A later
`pip install taichi` or `pip install -r` with `taichi==1.7.4` keeps it (a local label satisfies that pin). What replaces it is an
explicit request for the official wheel, such as `pip install --force-reinstall taichi==1.7.4`; afterwards `taichi._fork` is gone.
