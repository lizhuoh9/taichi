# Added in the fork lizhuoh9/taichi on 2026-11-07 (see FORK.md): tests of fast path d, descriptors for data_oriented.
"""Fast path d of taichi.lang.kernel_impl.data_oriented (FORK.md change 7) gives a data-oriented class descriptors
instead of Taichi 1.7.4's __getattribute__ hook. The risks: an attribute read, set or delete that gives another value
or error than the hook gave (a kernel, a method, a field, a property, a plain or class attribute, an instance attribute
of any name, a slot), a subclass that is read differently, a template key other than the hook's (a data-oriented
argument is keyed by a weak reference), state that does not survive ti.reset and ti.init, a kernel replaced or deleted
on a base after decoration that is still read in its old form or with its old _is_staticmethod, a class attribute that
introspects differently (callable(), inspect.signature, inspect.unwrap and attribute reads of what
inspect.getattr_static gives), an entry stored on an instance that is not read as the entry, an entry set later on a
class between the descriptor's class and the entry's that a read does not see (a downstream solver's tests count
launches that way: a count of 0 passed for nothing), a case the descriptors cannot repeat that is not handed back to
the hook (a base decorated after its decorated subclass among them), a
subclass made before its base was decorated that is read differently, a hierarchy handed back that keeps fast path d's
entries, and a difference from 1.7.4 other than those FORK.md lists.

One script makes every observation in a fresh interpreter on the CPU, with TI_LAUNCH_FASTPATH unset (on) and 0 (off);
off is Taichi 1.7.4's code. The observations must be equal, except the differences FORK.md lists, which the script
records apart and the tests check one by one; calls that succeed in 1.7.4 give the same values on.
"""

import json
import os
import subprocess
import sys

import pytest

CPU = "arch=ti.cpu, cpu_max_num_threads=1, offline_cache=False, log_level=ti.WARN"

SCENARIOS = r"""
import abc
import contextlib
import copy
import functools
import gc
import inspect
import json
import pickle
import re
import weakref
from unittest import mock

import pytest
import taichi as ti
from taichi.lang import kernel_impl
from taichi.lang.field import Field

ti.init(CPU_ARGUMENTS)
OBSERVED, DIFFERENCES = {}, {}


def describe(value):
    # A JSON description of a value that two processes compare.
    if isinstance(value, kernel_impl._BoundedDifferentiableMethod):
        return ["bound kernel", value.__name__, type(value._kernel_owner).__name__, value._is_staticmethod,
                value._primal.func.__qualname__]
    if inspect.ismethod(value):
        return ["method", value.__func__.__qualname__, type(value.__self__).__name__]
    if inspect.isfunction(value):
        return ["function", value.__qualname__, getattr(value, "_is_wrapped_kernel", False)]
    if isinstance(value, Field):
        return ["field", list(value.shape), str(value.dtype)]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return ["value", value]
    if isinstance(value, (list, tuple)):
        return ["sequence", [describe(item) for item in value]]
    return ["object", type(value).__name__]


def see(tag, thunk, into=OBSERVED):
    try:
        value = thunk()
    except Exception as error:  # an error is an observation too; an object's address differs between processes
        into[tag] = ["raises", type(error).__name__, re.sub(r"0x[0-9a-fA-F]+", "0x?", str(error).strip())]
    else:
        into[tag] = describe(value)


def data(obj):
    return obj.data.to_numpy().tolist()


def after(call, obj):
    call()
    return data(obj)


def keys(kernel):
    # The template keys of a kernel, a weak reference shown as whether its object is alive.
    return [[["weakref", element() is not None] if isinstance(element, weakref.ref) else repr(element)
             for element in key] for key in kernel._primal.mapper.mapping]


def details(*classes):
    return {cls.__name__: dict(hook="__getattribute__" in cls.__dict__, fast=cls in kernel_impl._FAST_DATA_ORIENTED,
                               descriptors=sorted(key for key, entry in cls.__dict__.items()
                                                  if isinstance(entry, kernel_impl._DataOrientedKernel)))
            for cls in classes}


# ------------------------------------------------------------------------- a decorated class with an undecorated base
class Recorder:
    # A data descriptor of its own: __set__ records the value, __get__ reads it back.
    def __set_name__(self, owner, name):
        self.key = "_" + name

    def __get__(self, obj, owner=None):
        if obj is None:
            return "Recorder"
        return obj.__dict__.get(self.key, "unset")

    def __set__(self, obj, value):
        obj.__dict__[self.key] = ["recorded", value]


class Mixin:
    MIXIN_CONST = 11
    recorded = Recorder()

    @property
    def mixin_settable(self):
        return self.__dict__.get("_mixin_settable", "unset")

    @mixin_settable.setter
    def mixin_settable(self, value):
        self.__dict__["_mixin_settable"] = ["mixin set", value]

    @ti.kernel
    def mixin_fill(self, value: ti.i32):
        for i in self.data:
            self.data[i] = value + i

    def mixin_method(self):
        return "mixin_method"

    @property
    def mixin_prop(self):
        return 21

    @property
    def mixin_kernel_by_property(self):
        return Mixin.mixin_fill  # the raw kernel, which the hook bound

    @functools.cached_property
    def cached(self):
        return [self.n, "cached"]


@ti.data_oriented
class Top(Mixin):
    CONST = 3
    layer_recorded = Recorder()

    @property
    def settable(self):
        return self.__dict__.get("_settable", "unset")

    @settable.setter
    def settable(self, value):
        self.__dict__["_settable"] = ["set", value]

    def __init__(self, n):
        self.n = n
        self.data = ti.field(ti.i32, shape=n)
        self.items = [1, 2]

    @ti.kernel
    def fill(self, value: ti.i32):
        for i in self.data:
            self.data[i] = value * (i + 1)

    @ti.kernel
    def first(self) -> ti.i32:
        return self.data[0] + 10 * self.data[1]

    @staticmethod
    @ti.kernel
    def answer() -> ti.i32:
        return 42

    @ti.kernel
    def use(self, other: ti.template()):
        for i in self.data:
            self.data[i] = other.data[i] + 1

    def method(self):
        return "method"

    @property
    def prop(self):
        return self.n * 2

    @property
    def fill_by_property(self):
        return self.fill


r = Top(4)
for name in ("fill", "first", "answer", "use", "mixin_fill", "method", "mixin_method", "prop", "mixin_prop",
             "fill_by_property", "mixin_kernel_by_property", "cached", "cached", "data", "n", "items", "CONST",
             "MIXIN_CONST", "missing"):
    see("r." + name, lambda name=name: getattr(r, name))
for name in ("fill", "answer", "mixin_fill", "method", "CONST", "prop", "mixin_prop", "missing"):
    see("Top." + name, lambda name=name: getattr(Top, name))
see("Mixin.mixin_fill", lambda: Mixin.mixin_fill)
see("r.fill is r.fill", lambda: r.fill is r.fill)
see("r.data is r.data", lambda: r.data is r.data)
see("r.fill(2)", lambda: after(lambda: r.fill(2), r))
see("r.first()", lambda: r.first())
see("r.answer()", lambda: r.answer())
see("r.mixin_fill(5)", lambda: after(lambda: r.mixin_fill(5), r))
see("r.fill_by_property(3)", lambda: after(lambda: r.fill_by_property(3), r))
see("r.mixin_kernel_by_property(6)", lambda: after(lambda: r.mixin_kernel_by_property(6), r))
see("r.fill('x')", lambda: r.fill("x"))
see("r.fill(1, 2)", lambda: r.fill(1, 2))
see("flags after reads", lambda: [getattr(Top.fill, "_is_staticmethod", "unset"),
                                  getattr(Top.answer, "_is_staticmethod", "unset"),
                                  getattr(Mixin.mixin_fill, "_is_staticmethod", "unset")])

# -------------------------------------------------------------------------------------- set and delete on instances
r.CONST = 5
see("r.CONST shadowed", lambda: r.CONST)
r.fill = 7
see("r.fill shadowed", lambda: r.fill)
see("r.__dict__", lambda: sorted(r.__dict__))
del r.fill
see("r.fill after del", lambda: r.fill)


def delete(obj, name):
    delattr(obj, name)


see("del r.fill again", lambda: delete(r, "fill"))
see("del r.missing", lambda: delete(r, "missing"))
see("del r.prop", lambda: delete(r, "prop"))
see("del r.mixin_prop", lambda: delete(r, "mixin_prop"))
r.fill = Top.fill
see("r.fill = Top.fill", lambda: r.fill)
see("r.fill(4) through the instance attribute", lambda: after(lambda: r.fill(4), r))
del r.fill
r.n = 9
see("r.n set", lambda: r.n)
see("r.prop after n", lambda: r.prop)
see("r.prop = 1", lambda: setattr(r, "prop", 1))
see("r.mixin_prop = 1", lambda: setattr(r, "mixin_prop", 1))
see("r.cached after n", lambda: r.cached)
for name in ("settable", "mixin_settable", "recorded", "layer_recorded"):
    see("r." + name + " before", lambda name=name: getattr(r, name))
    see("r." + name + " = 4", lambda name=name: (setattr(r, name, 4), getattr(r, name))[1])
    see("Top." + name, lambda name=name: getattr(Top, name))
see("r.__dict__ after the setters", lambda: sorted(r.__dict__))
see("del r.settable", lambda: delete(r, "settable"))
see("del r.recorded", lambda: delete(r, "recorded"))
del r.items
see("r.items after del", lambda: r.items)
see("del r.items again", lambda: delete(r, "items"))

# ---------------------------------------------------------- kernels and other values kept on an instance (1.7.4 binds)
r6 = Top(2)
r6.other = Top.fill
see("class kernel under another name", lambda: r6.other)
see("class kernel under another name, called", lambda: after(lambda: r6.other(3), r6))


@ti.kernel
def module_kernel(x: ti.template()):
    for i in x:
        x[i] = 2


r6.hook = module_kernel
see("module kernel read", lambda: r6.hook)
see("module kernel flag", lambda: getattr(module_kernel, "_is_staticmethod", "unset"))
see("module kernel, called", lambda: after(lambda: r6.hook(r6.data), r6))
r6.mocked = mock.Mock()
see("mock value", lambda: r6.mocked)
r7 = Top(2)
see("another instance, other", lambda: r7.other)
see("another instance, mocked", lambda: r7.mocked)
r6.other = 3
see("an int written over the stored kernel", lambda: r6.other)
del r6.other
see("the stored kernel deleted", lambda: r6.other)

# ------------------------------------------------------------------- a data-oriented object as a template argument
r2 = Top(4)
r2.fill(1)
see("r.use(r2)", lambda: after(lambda: r.use(r2), r))
see("use keys", lambda: keys(Top.use))
del r2
gc.collect()
see("use keys after del", lambda: keys(Top.use))

# ------------------------------------------------------------------------------------------------- subclasses
@ti.data_oriented
class Child(Top):
    @ti.kernel
    def child_fill(self):
        for i in self.data:
            self.data[i] = 100 + i

    @ti.kernel
    def fill(self, value: ti.i32):
        for i in self.data:
            self.data[i] = -value


class Plain(Top):
    @ti.kernel
    def plain_fill(self):
        for i in self.data:
            self.data[i] = 7


class Hooked(Top):
    def __init_subclass__(cls, **kwargs):
        cls.marker = "hooked"


class Deep(Hooked):
    @ti.kernel
    def deep_fill(self):
        for i in self.data:
            self.data[i] = 9


class KwBase(Top):
    def __init_subclass__(cls, tag=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.tag = tag


class KwSub(KwBase, tag="tagged"):
    @ti.kernel
    def kw_fill(self):
        for i in self.data:
            self.data[i] = 11


class Sibling:
    def __init_subclass__(cls, **kwargs):  # no super(): the next __init_subclass__ in the MRO is never called
        cls.sibling = True


class Bypass(Sibling, Top):
    @ti.kernel
    def bypass_fill(self):
        for i in self.data:
            self.data[i] = 13


c, p, d, k, b = Child(3), Plain(3), Deep(3), KwSub(3), Bypass(3)
for tag, obj in (("c", c), ("p", p), ("d", d), ("k", k), ("b", b)):
    for name in ("fill", "child_fill", "plain_fill", "deep_fill", "kw_fill", "bypass_fill", "mixin_fill", "answer",
                 "prop", "data", "marker", "tag", "sibling"):
        see(tag + "." + name, lambda obj=obj, name=name: getattr(obj, name))
see("c.child_fill()", lambda: after(c.child_fill, c))
see("c.fill(2)", lambda: after(lambda: c.fill(2), c))
see("c.mixin_fill(1)", lambda: after(lambda: c.mixin_fill(1), c))
see("p.plain_fill()", lambda: after(p.plain_fill, p))
see("p.fill(3)", lambda: after(lambda: p.fill(3), p))
see("d.deep_fill()", lambda: after(d.deep_fill, d))
see("d.answer()", lambda: d.answer())
see("k.kw_fill()", lambda: after(k.kw_fill, k))
see("b.bypass_fill()", lambda: after(b.bypass_fill, b))
see("kwargs refused by object", lambda: type("Refused", (Top,), {}, tag="x"))

# ------------------------------------------------------------------ a diamond, a metaclass, double decoration
class Base0:
    @ti.kernel
    def base_fill(self):
        for i in self.data:
            self.data[i] = 21


class DA(Base0):
    @ti.kernel
    def da_fill(self):
        for i in self.data:
            self.data[i] = 22


class DB(Base0):
    @ti.kernel
    def base_fill(self):
        for i in self.data:
            self.data[i] = 23


@ti.data_oriented
class Diamond(DA, DB):
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)


dm = Diamond()
see("dm.base_fill()", lambda: after(dm.base_fill, dm))
see("dm.da_fill()", lambda: after(dm.da_fill, dm))


@ti.data_oriented
class WithMeta(abc.ABC):
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 31


wm = WithMeta()
see("wm.fill()", lambda: after(wm.fill, wm))


class TwiceBody:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 41


Twice = ti.data_oriented(ti.data_oriented(TwiceBody))
tw = Twice()
see("tw.fill()", lambda: after(tw.fill, tw))


@ti.data_oriented
class WithClassMethod:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @classmethod
    @ti.kernel
    def ck(cls) -> ti.i32:
        return 51

    hook = module_kernel


wc = WithClassMethod()
see("wc.ck", lambda: wc.ck)
see("wc.ck()", lambda: wc.ck())
see("wc.hook", lambda: wc.hook)
see("wc.hook(wc.data)", lambda: after(lambda: wc.hook(wc.data), wc))


# ------------------------------------------------- static introspection: inspect.getattr_static, signature, unwrap
def introspect(owner, name):
    # What inspect.getattr_static gives for name: whether it is callable, its signature, and what unwrap reaches.
    found = inspect.getattr_static(owner, name)
    try:
        signature = re.sub(r"0x[0-9a-fA-F]+", "0x?", str(inspect.signature(found)))
    except TypeError as error:
        signature = ["TypeError", str(error).endswith(" is not a callable object")]
    unwrapped = inspect.unwrap(found)
    return [callable(found), signature, getattr(unwrapped, "__qualname__", type(unwrapped).__name__)]


for owner, name in ((Top, "fill"), (Top, "first"), (Top, "answer"), (Top, "use"), (Top, "mixin_fill"),
                    (Top, "method"), (Top, "prop"), (Top, "mixin_prop"), (Top, "mixin_kernel_by_property"),
                    (Top, "cached"), (Top, "recorded"), (Top, "layer_recorded"), (Mixin, "mixin_fill"),
                    (Child, "fill"), (Child, "child_fill"), (Child, "first"), (Child, "mixin_fill"),
                    (Plain, "plain_fill"), (Plain, "fill"), (WithClassMethod, "ck"), (WithClassMethod, "hook"),
                    (r7, "fill"), (r7, "answer"), (r7, "mixin_fill")):
    label = owner.__name__ if isinstance(owner, type) else "an instance of " + type(owner).__name__
    see("introspect " + label + "." + name, lambda owner=owner, name=name: introspect(owner, name))
see("a call of Top.answer as getattr_static gives it", lambda: inspect.getattr_static(Top, "answer")())
see("a call of WithClassMethod.hook as getattr_static gives it",
    lambda: after(lambda: inspect.getattr_static(WithClassMethod, "hook")(wc.data), wc))


# ------------------------------------------------------- attribute reads of what inspect.getattr_static gives
ATTRIBUTES = ("__name__", "__qualname__", "__module__", "__doc__", "__globals__", "__code__", "__defaults__",
              "__kwdefaults__", "__annotations__", "__closure__", "__dict__", "__func__", "__self__",
              "__isabstractmethod__", "__objclass__", "__text_signature__", "__signature__", "fget", "fset", "fdel",
              "func", "attrname", "key", "_is_wrapped_kernel", "_is_classkernel", "_is_staticmethod", "_primal",
              "_adjoint", "grad", "no_such_attribute")


def attribute_reads(owner, name):
    # Each attribute of what getattr_static gives for name, as JSON that two processes compare.
    found = inspect.getattr_static(owner, name)
    seen = {"getdoc": inspect.getdoc(found), "getmodule": getattr(inspect.getmodule(found), "__name__", None)}
    for attribute in ATTRIBUTES:
        try:
            value = getattr(found, attribute)
        except Exception as error:
            seen[attribute] = [type(error).__name__, re.sub(r"0x[0-9a-fA-F]+", "0x?", str(error))]
            continue
        if attribute == "__globals__":
            value = value["__name__"]
        elif attribute == "__code__":
            value = value.co_name
        elif isinstance(value, dict):
            value = sorted(value)
        elif not isinstance(value, (bool, int, str, type(None))):
            value = type(value).__name__
        seen[attribute] = value
    return json.dumps(seen, sort_keys=True)


@ti.data_oriented
class TopDoc:
    @ti.kernel
    def with_doc(self):
        "A kernel's docstring."

    @ti.kernel
    def without_doc(self):
        pass


class SubDoc(TopDoc):
    @ti.kernel
    def with_doc(self):
        pass


found = inspect.getattr_static(TopDoc, "with_doc")
for owner, name in ((Top, "fill"), (Top, "answer"), (Top, "mixin_fill"), (Top, "mixin_prop"), (Top, "cached"),
                    (Top, "recorded"), (Top, "layer_recorded"), (Top, "mixin_kernel_by_property"), (Child, "fill"),
                    (Child, "first"), (Child, "child_fill"), (Plain, "plain_fill"), (WithClassMethod, "ck"),
                    (WithClassMethod, "hook"), (r7, "fill"), (TopDoc, "with_doc"), (TopDoc, "without_doc")):
    label = owner.__name__ if isinstance(owner, type) else "an instance of " + type(owner).__name__
    see("attribute reads of " + label + "." + name, lambda owner=owner, name=name: attribute_reads(owner, name))
see("a copy of what getattr_static gives: callable, its __name__", lambda: [callable(copy.copy(found)),
                                                                            copy.copy(found).__name__])


def unset_slots():
    # A descriptor whose slots are not set answers no attribute: a plain read of self.raw would recurse.
    blank = object.__new__(kernel_impl._DataOrientedKernel)
    return [hasattr(blank, "raw"), hasattr(blank, "any_other_name")]


see("a descriptor whose slots are not set", unset_slots)
rd = TopDoc()
see("what getattr_static gives: its type", lambda: type(found).__name__, DIFFERENCES)
see("what getattr_static gives: isfunction, isroutine", lambda: [inspect.isfunction(found), inspect.isroutine(found)],
    DIFFERENCES)
see("what getattr_static gives: repr", lambda: repr(found).split(" ")[0], DIFFERENCES)
see("what getattr_static gives: dir", lambda: "__code__" in dir(found), DIFFERENCES)
see("what getattr_static gives: the class read", lambda: [found is TopDoc.with_doc, found == TopDoc.with_doc],
    DIFFERENCES)
see("what getattr_static gives: a subclass's for an inherited kernel",
    lambda: inspect.getattr_static(Child, "mixin_fill") is inspect.getattr_static(Top, "mixin_fill"), DIFFERENCES)
see("what getattr_static gives: an attribute set on it", lambda: setattr(found, "tag", 1), DIFFERENCES)
see("what getattr_static gives: __wrapped__", lambda: found.__wrapped__ is TopDoc.with_doc, DIFFERENCES)
see("what getattr_static gives: __get__", lambda: found.__get__(rd, TopDoc), DIFFERENCES)
see("what getattr_static gives: getfile", lambda: inspect.getfile(found).endswith("kernel_impl.py"), DIFFERENCES)
see("what getattr_static gives: getdoc of a kernel that inherits its docstring",
    lambda: inspect.getdoc(inspect.getattr_static(SubDoc, "with_doc")), DIFFERENCES)

# ---------------------------------------------- the cases handed back to the hook, each in a hierarchy of its own
@ti.data_oriented
class TopP:
    def __init__(self):
        self.n = 2
        self.data = ti.field(ti.i32, shape=2)

    @property
    def prop(self):
        return self.n * 2

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 5


@ti.data_oriented
class ChildP(TopP):
    @ti.kernel
    def child_fill(self):
        for i in self.data:
            self.data[i] = 6


class OverridesProp(ChildP):
    prop = 99


o, q, cp = OverridesProp(), TopP(), ChildP()
see("o.prop", lambda: o.prop)
see("q.prop", lambda: q.prop)
see("cp.child_fill()", lambda: after(cp.child_fill, cp))
see("o.fill()", lambda: after(o.fill, o))


@ti.data_oriented
class TopG:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 6


class Logging(TopG):
    def __getattribute__(self, item):
        return super().__getattribute__(item)


g = Logging()
see("g.fill", lambda: g.fill)
see("g.fill()", lambda: after(g.fill, g))


@ti.data_oriented
class TopZ:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 61


class SetsItself(TopZ):
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


z = SetsItself()
z.other = TopZ.fill
z.n = 3
see("z.other", lambda: z.other)
see("z.n", lambda: z.n)
see("z.fill()", lambda: after(z.fill, z))


@ti.data_oriented
class TopX:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def x_fill(self):
        for i in self.data:
            self.data[i] = 71


@ti.data_oriented
class TopY:
    @ti.kernel
    def y_fill(self):
        for i in self.data:
            self.data[i] = 72


class XY(TopX, TopY):
    pass


xy = XY()
see("xy.x_fill()", lambda: after(xy.x_fill, xy))
see("xy.y_fill()", lambda: after(xy.y_fill, xy))


@ti.data_oriented
class KernelProp:
    @property
    @ti.kernel
    def value(self) -> ti.i32:
        return 42


see("KernelProp().value", lambda: KernelProp().value)


class MixinKP:
    @property
    @ti.kernel
    def value(self) -> ti.i32:
        return 43


@ti.data_oriented
class TopKP(MixinKP):
    pass


see("TopKP().value", lambda: TopKP().value)


@ti.data_oriented
class WriteOnly:
    def _set(self, value):
        self.stored = value

    only = property(fset=_set)


wo = WriteOnly()
see("wo.only", lambda: wo.only)
see("wo.only = 3", lambda: (setattr(wo, "only", 3), wo.stored)[1])


@ti.data_oriented
class Slotted:
    __slots__ = ("data", "__weakref__")

    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 3


sl = Slotted()
see("sl.fill", lambda: sl.fill)
see("sl.fill()", lambda: after(sl.fill, sl))


@ti.data_oriented
class TopG2:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 62


class LoggingDirect(TopG2):
    def __getattribute__(self, item):
        return object.__getattribute__(self, item)  # skips TopG2's hook


g2 = LoggingDirect()
see("g2.fill through a __getattribute__ that skips the hook", lambda: g2.fill)
see("g2.fill() through a __getattribute__ that skips the hook", lambda: after(g2.fill, g2))


@ti.data_oriented
class TopX2:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @staticmethod
    @ti.kernel
    def sx() -> ti.i32:
        return 73


@ti.data_oriented
class TopY2:
    pass


class XY2(TopX2, TopY2):
    pass


xy2 = XY2()
see("xy2.sx, a staticmethod kernel under two hooks", lambda: xy2.sx)  # TopY2's hook bound it, with its own flag
see("xy2.sx() under two hooks", lambda: xy2.sx())
HANDED_BACK = (TopP, ChildP, OverridesProp, TopG, Logging, TopZ, SetsItself, TopX, TopY, XY, TopG2, LoggingDirect,
               TopX2, TopY2, XY2)
see("class dicts of the hierarchies handed back to the hook",
    lambda: {cls.__name__: sorted(vars(cls)) for cls in HANDED_BACK})


# -------------------------------------------------------------------- a base decorated after a subclass was made
class BaseV1:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @staticmethod
    @ti.kernel
    def st_u() -> ti.i32:
        return 1

    @ti.kernel
    def ku(self) -> ti.i32:
        return self.data[0] + 2


@ti.data_oriented
class DecoratedFirst(BaseV1):
    @staticmethod
    @ti.kernel
    def st_v() -> ti.i32:
        return 3

    @ti.kernel
    def kv(self) -> ti.i32:
        return self.data[0] + 4


class BelowDecoratedFirst(DecoratedFirst):
    @ti.kernel
    def kw(self) -> ti.i32:
        return self.data[0] + 5


ti.data_oriented(BaseV1)
df, bdf = DecoratedFirst(), BelowDecoratedFirst()
for name in ("st_u", "ku", "st_v", "kv"):  # BaseV1's hook, the innermost, bound each with BaseV1's flag
    see("df." + name + ", its base decorated later", lambda name=name: getattr(df, name))
    see("df." + name + "(), its base decorated later", lambda name=name: getattr(df, name)())
see("bdf.kw, a base of its base decorated later", lambda: bdf.kw)
see("bdf.kw(), a base of its base decorated later", lambda: bdf.kw())
see("class dicts after a base was decorated later",
    lambda: {cls.__name__: sorted(vars(cls)) for cls in (BaseV1, DecoratedFirst, BelowDecoratedFirst)})


class BaseV2:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def kb(self) -> ti.i32:
        return self.data[0] + 6


class UndecoratedFirst(BaseV2):
    @ti.kernel
    def ku2(self) -> ti.i32:
        return self.data[0] + 7

    @staticmethod
    @ti.kernel
    def su2() -> ti.i32:
        return 8


class BelowUndecoratedFirst(UndecoratedFirst):
    @ti.kernel
    def kb2(self) -> ti.i32:
        return self.data[0] + 9


before = UndecoratedFirst()  # an instance made, and a kernel stored on it, before the base was decorated
before.stored = BaseV2.kb
ti.data_oriented(BaseV2)
uf, buf = UndecoratedFirst(), BelowUndecoratedFirst()
see("before.ku2, an instance made before its base was decorated", lambda: before.ku2)
see("before.ku2()", lambda: before.ku2())
see("a kernel stored on an instance before its class was decorated", lambda: before.stored, DIFFERENCES)
for name in ("kb", "ku2", "su2"):
    see("uf." + name + ", its base decorated later", lambda name=name: getattr(uf, name))
    see("uf." + name + "(), its base decorated later", lambda name=name: getattr(uf, name)())
see("buf.kb2", lambda: buf.kb2)
see("buf.kb2()", lambda: buf.kb2())
see("introspect UndecoratedFirst.ku2", lambda: introspect(UndecoratedFirst, "ku2"))


class BaseV3:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def kx(self) -> ti.i32:
        return self.data[0] + 10


class ReadsFirst(BaseV3):
    def __getattribute__(self, item):
        return super().__getattribute__(item)


ti.data_oriented(BaseV3)
rf, base3 = ReadsFirst(), BaseV3()
see("rf.kx, another __getattribute__ made before its base was decorated", lambda: rf.kx)
see("rf.kx()", lambda: rf.kx())
see("base3.kx", lambda: base3.kx)


class BaseV4:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @staticmethod
    @ti.kernel
    def s4() -> ti.i32:
        return 11


@ti.data_oriented
class HookedFirst(BaseV4):
    only = property(fset=lambda self, value: None)  # its own property without a getter: HookedFirst has the hook

    @staticmethod
    @ti.kernel
    def s5() -> ti.i32:
        return 12


ti.data_oriented(BaseV4)
hf = HookedFirst()
for name in ("s4", "s5"):
    see("hf." + name + ", its subclass on the hook when the base was decorated", lambda name=name: getattr(hf, name))
    see("hf." + name + "()", lambda name=name: getattr(hf, name)())
see("class dicts after a base of a class on the hook was decorated",
    lambda: {cls.__name__: sorted(vars(cls)) for cls in (BaseV3, ReadsFirst, BaseV4, HookedFirst)})


# ------------------------------------------------------------------------------- a slot next to an instance __dict__
@ti.data_oriented
class WithSlots:
    __slots__ = ("held", "__dict__", "__weakref__")

    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 4


ws = WithSlots()
see("a slot before it is set", lambda: ws.held)
ws.held = WithSlots.fill
see("a kernel kept in a slot", lambda: ws.held)
see("a kernel kept in a slot, called", lambda: after(ws.held, ws))
see("WithSlots.held", lambda: WithSlots.held)
see("introspect WithSlots.held", lambda: introspect(WithSlots, "held"))
del ws.held
see("a slot deleted", lambda: ws.held)
see("a slot deleted again", lambda: delete(ws, "held"))
see("attribute reads of WithSlots.held", lambda: attribute_reads(WithSlots, "held"))


# --------------------------------------------------------------------------------- a getset of the interpreter
@ti.data_oriented
class TopE(Exception):
    @ti.kernel
    def fill(self):
        pass


e = TopE(1, 2)
see("a getset read", lambda: e.args)
see("a getset: the class read", lambda: type(TopE.args).__name__)
see("introspect TopE.args", lambda: introspect(TopE, "args"))
see("attribute reads of TopE.args", lambda: attribute_reads(TopE, "args"))
e.args = (3,)
see("a getset after a set", lambda: [e.args, str(e)])
see("a getset deleted", lambda: delete(e, "args"))
see("a getset set to what it refuses", lambda: setattr(e, "args", 5))
see("a getset after those", lambda: e.args)


@ti.data_oriented
class TopInt(int):
    pass


see("a read-only getset", lambda: TopInt(5).real)
see("a read-only getset set", lambda: setattr(TopInt(5), "real", 1))
see("a read-only getset deleted", lambda: delete(TopInt(5), "real"))
see("a plain attribute of an int subclass", lambda: (lambda v: (setattr(v, "n", 2), v.n)[1])(TopInt(5)))

# ------------------------------------------------------------- kernels of a base replaced, deleted or added later
class MixinM:
    @ti.kernel
    def mk(self, value: ti.i32):
        for i in self.data:
            self.data[i] = value

    @ti.kernel
    def gone(self):
        for i in self.data:
            self.data[i] = 1


@ti.data_oriented
class TopM(MixinM):
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def own(self):
        for i in self.data:
            self.data[i] = 81


m = TopM()
original = MixinM.mk
MixinM.mk = lambda self, value: ["python", value]
see("m.mk replaced", lambda: m.mk)
see("m.mk(3) replaced", lambda: m.mk(3))
see("introspect TopM.mk while its base's entry is replaced", lambda: introspect(TopM, "mk"), DIFFERENCES)
MixinM.mk = original
see("m.mk restored", lambda: m.mk)
see("m.mk(4) restored", lambda: after(lambda: m.mk(4), m))
del MixinM.gone
see("m.gone deleted", lambda: m.gone)
see("introspect TopM.gone after its base's entry is deleted", lambda: introspect(TopM, "gone"), DIFFERENCES)
own = TopM.__dict__["own"]
TopM.own = lambda self: "python own"
see("m.own replaced", lambda: m.own())
TopM.own = own  # put back as pytest's monkeypatch does: the class __dict__ entry
see("m.own put back", lambda: m.own)
MixinM.alias = original
see("m.alias(5)", lambda: after(lambda: m.alias(5), m))


class SubM(TopM):
    pass


sm = SubM()
see("sm.alias", lambda: sm.alias)
see("sm.alias(6)", lambda: after(lambda: sm.alias(6), sm))


class MixA:
    pass


class MixB:
    @ti.kernel
    def kk(self):
        for i in self.data:
            self.data[i] = 91


@ti.data_oriented
class SolverAB(MixA, MixB):
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)


ab = SolverAB()
MixA.kk = lambda self: "MixA.kk"  # earlier in SolverAB's MRO than MixB, which defined the kernel


# --------------------------------- the _is_staticmethod of a read after the entry the descriptor read was replaced
@ti.data_oriented
class TopF:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @staticmethod
    @ti.kernel
    def flip() -> ti.i32:
        return 1

    @ti.kernel
    def plain(self) -> ti.i32:
        return self.data[0] + 2


class SubF(TopF):
    pass


sf, sf2, sf3 = SubF(), SubF(), SubF()
sf.data[0], sf2.data[0], sf3.data[0] = 5, 6, 7
sf3.later = TopF.plain  # a name an instance stored a kernel under
see("flip before", lambda: sf.flip)
see("flip() before", lambda: sf.flip())
see("later before", lambda: sf3.later)
TopF.flip = TopF.plain  # the first decorated class's staticmethod kernel replaced by a plain kernel
TopF.later = staticmethod(TopF.plain)  # and a staticmethod set on it under the name sf3 stored a kernel under
sf2.flip = TopF.plain
see("flip after the replacement", lambda: sf.flip)
see("flip() after the replacement", lambda: sf.flip())
see("flip kept on an instance after the replacement", lambda: sf2.flip)
see("flip kept on an instance after the replacement, called", lambda: sf2.flip())
see("later after a staticmethod was set under it", lambda: sf3.later)

# ------------------------------------------------------------------------------ a class that is not decorated
class NotDecorated:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 1


see("NotDecorated().fill()", lambda: NotDecorated().fill())

# --------------------------------------------------------------------------------------- ti.reset and ti.init
ti.reset()
ti.init(CPU_ARGUMENTS)
r5 = Top(3)
see("after reset r5.fill(4)", lambda: after(lambda: r5.fill(4), r5))
see("after reset r5.first()", lambda: r5.first())
see("after reset r5.answer()", lambda: r5.answer())
see("after reset c5.child_fill()", lambda: (lambda c5: after(c5.child_fill, c5))(Child(2)))

# ------------------------------------------------------------- the differences FORK.md lists, observed in both modes
see("alias value", lambda: m.alias, DIFFERENCES)
see("own put back, class-level read", lambda: (setattr(TopM, "own", TopM.own), m.own)[1], DIFFERENCES)
ab2 = SolverAB()
see("a kernel name set on an earlier class", lambda: ab2.kk)


@ti.data_oriented
class TopS:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 8


class SubS(TopS):
    def call_super(self):
        return super().fill()

    def read_super(self):
        return super().fill


see("super call", lambda: (lambda s: after(s.call_super, s))(SubS()), DIFFERENCES)
shadowing = SubS()
shadowing.fill = 5
see("super read of a shadowed name", lambda: shadowing.read_super(), DIFFERENCES)
see("class-level call", lambda: (lambda s: after(lambda: TopS.fill(s), s))(TopS()), DIFFERENCES)
r8 = Top(2)
vars(r8)["stored"] = Top.fill
see("a kernel put into vars()", lambda: r8.stored, DIFFERENCES)
r10 = Top(2)
object.__setattr__(r10, "via_object", Top.fill)
see("a kernel set through object.__setattr__", lambda: r10.via_object, DIFFERENCES)
see("a call of the descriptor that getattr_static gives",
    lambda: (lambda s: after(lambda: inspect.getattr_static(TopS, "fill")(s), s))(TopS()), DIFFERENCES)


# the class __dict__: getattr_static of names an instance stored a kernel under, dir, classify_class_attrs
@ti.data_oriented
class TopI:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self, value: ti.i32):
        for i in self.data:
            self.data[i] = value

    @staticmethod
    @ti.kernel
    def answer() -> ti.i32:
        return 4

    def method(self):
        return "method"


ri = TopI()
ri.stored = TopI.fill  # a name TopI has no entry for
ri.method = TopI.fill  # a name TopI has an entry for
see("introspect TopI.method after an instance stored a kernel under it", lambda: introspect(TopI, "method"))
see("TopI.method(ri) after an instance stored a kernel under it", lambda: TopI.method(ri))
see("ri.method, a kernel stored under the name of a method", lambda: ri.method)
see("introspect TopI.answer", lambda: introspect(TopI, "answer"))
see("introspect TopI.stored, a name an instance stored a kernel under", lambda: introspect(TopI, "stored"),
    DIFFERENCES)
see("introspect ri.stored", lambda: introspect(ri, "stored"), DIFFERENCES)
ri.fill = 7
see("introspect ri.fill, an instance value over a kernel", lambda: introspect(ri, "fill"), DIFFERENCES)
see("TopI's class __dict__ and dir",
    lambda: [sorted(vars(TopI)), type(vars(TopI)["fill"]).__name__, inspect.isdatadescriptor(vars(TopI)["fill"]),
             sorted(set(dir(TopI)) - set(dir(object)))], DIFFERENCES)


def classified(cls, names):
    return sorted([a.name, a.kind, a.defining_class.__name__] for a in inspect.classify_class_attrs(cls)
                  if a.name in names)


see("classify_class_attrs", lambda: [classified(Top, ("fill", "mixin_fill", "answer", "mixin_prop", "method")),
                                     classified(WithClassMethod, ("ck",))], DIFFERENCES)


# entries set on a class after the class was made
class MixinT:
    @ti.kernel
    def mixin_peek(self) -> ti.i32:
        return self.data[0] + 100


@ti.data_oriented
class TopT(MixinT):
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @staticmethod
    @ti.kernel
    def twice(x: ti.i32) -> ti.i32:
        return 2 * x

    @staticmethod
    @ti.kernel
    def peek(holder: ti.template()) -> ti.i32:
        return holder.data[0]

    @ti.kernel
    def plain(self) -> ti.i32:
        return self.data[0] + 1


class SubT(TopT):
    early = 1


@ti.data_oriented
class ChildT(TopT):
    @ti.kernel
    def own_k(self) -> ti.i32:
        return 3


rt, st, ct, hooked = TopT(), SubT(), ChildT(), Logging()  # Logging's TopG went back to the 1.7.4 hook
rt.data[0], st.data[0] = 4, 5
TopT.s_twice = staticmethod(TopT.twice)
TopT.s_peek = staticmethod(TopT.peek)
TopT.c_plain = classmethod(TopT.plain)
TopT.mocked = mock.Mock()
TopT.plain = mock.Mock()  # in place of TopT's own kernel
MixinT.returns_kernel = property(lambda self: MixinT.mixin_peek)
MixinT.is_kernel = property(MixinT.mixin_peek)
TopT.no_getter = property(fset=lambda self, value: None)
TopT.early = property(lambda self: "property")
TopT.own_k = staticmethod(TopT.twice)  # on the first decorated class, under the name of ChildT's kernel
see("a staticmethod kernel set later", lambda: st.s_twice, DIFFERENCES)
see("a staticmethod kernel set later, called with a scalar", lambda: st.s_twice(3), DIFFERENCES)
see("a staticmethod kernel set later, called without arguments", lambda: st.s_twice(), DIFFERENCES)
see("a staticmethod kernel set later, called with an instance under the hook", lambda: st.s_peek(hooked),
    DIFFERENCES)
see("a staticmethod kernel set later, called with an instance on the fast path", lambda: st.s_peek(st))
see("a classmethod kernel set later", lambda: st.c_plain, DIFFERENCES)
see("a classmethod kernel set later, called", lambda: st.c_plain(), DIFFERENCES)
see("a Mock set later under a new name", lambda: st.mocked, DIFFERENCES)
see("a Mock set later in place of the class's own kernel", lambda: rt.plain, DIFFERENCES)
see("a Mock set later in place of a base's kernel, read through the subclass's descriptor", lambda: st.plain)
see("a property set later on an undecorated class, its getter returning a kernel", lambda: st.returns_kernel,
    DIFFERENCES)
see("a property set later on an undecorated class, its getter a kernel", lambda: st.is_kernel, DIFFERENCES)
see("a property without a getter set later on a decorated class", lambda: st.no_getter, DIFFERENCES)
see("a property set later on a decorated class, its name defined before it", lambda: st.early, DIFFERENCES)
see("a staticmethod set later on the first decorated class under a subclass's kernel name", lambda: ct.own_k,
    DIFFERENCES)


# a subclass whose __init_subclass__ chain skips fast path d's hook, before and after its first set
class SiblingD:
    def __init_subclass__(cls, **kwargs):  # no super()
        cls.sibling = True


@ti.data_oriented
class TopD:
    @ti.kernel
    def root_d(self) -> ti.i32:
        return 1


class BypassD(SiblingD, TopD):
    @ti.kernel
    def bypass_d(self) -> ti.i32:
        return 2

    @property
    @ti.kernel
    def by_property(self) -> ti.i32:
        return 3


bd = object.__new__(BypassD)  # no attribute set yet
see("a subclass that skipped the hook, before its first set", lambda: bd.bypass_d, DIFFERENCES)
see("its property whose getter is a kernel, before the first set", lambda: bd.by_property, DIFFERENCES)
bd.n = 1  # the first set: its check hands TopD's hierarchy back to the hook (the property)
see("a subclass that skipped the hook, after its first set", lambda: bd.bypass_d)
see("its property whose getter is a kernel, after the first set", lambda: bd.by_property)


# an entry set later on a class between the descriptor's class and the entry's: a kernel, a function, None, a property
class BaseB:
    @ti.kernel
    def kb(self) -> ti.i32:
        return 31

    @ti.kernel
    def other(self) -> ti.i32:
        return 32


class MidB(BaseB):
    pass


@ti.data_oriented
class TopB(MidB):
    pass


rb = TopB()
see("a kernel before anything is set on a middle class", lambda: rb.kb)
MidB.kb = BaseB.other
see("a kernel set on a middle class", lambda: rb.kb)
see("a kernel set on a middle class, called", lambda: rb.kb())
see("a kernel set on a middle class, the class read", lambda: TopB.kb)
MidB.kb = lambda self: "plain"
see("a plain function set on a middle class", lambda: rb.kb)
see("a plain function set on a middle class, called", lambda: rb.kb())
MidB.kb = None
see("None set on a middle class", lambda: rb.kb)
see("None set on a middle class, the class read", lambda: TopB.kb)
rb2 = TopB()
rb2.kb = 5
see("None set on a middle class, an instance value", lambda: rb2.kb)
MidB.kb = property(lambda self: "property")
see("a property set on a middle class, over an instance value", lambda: rb2.kb)
del MidB.kb
see("the new entry deleted, the old kernel", lambda: rb.kb)
see("the new entry deleted, called", lambda: rb.kb())
see("the new entry deleted, an instance value", lambda: rb2.kb)


# A downstream solver's way of counting a mixin's kernel launches: getattr, set, read and call, restore, delattr, read
@contextlib.contextmanager
def counting(cls, names):
    counts = dict.fromkeys(names, 0)
    originals = {name: getattr(cls, name) for name in names}
    for name in names:
        setattr(cls, "_counted_orig_" + name, originals[name])

        def counted(self, *args, _name=name):
            counts[_name] += 1
            return getattr(self, "_counted_orig_" + _name)(*args)

        setattr(cls, name, counted)
    try:
        yield counts
    finally:
        for name in names:
            setattr(cls, name, originals[name])
            delattr(cls, "_counted_orig_" + name)


class StorageC:
    @ti.kernel
    def validate(self) -> ti.i32:
        return 51


class ProjectionC(StorageC):
    def project(self):
        return self.validate()


@ti.data_oriented
class SolverC(ProjectionC):
    pass


sc = SolverC()
with counting(ProjectionC, ("validate",)) as seen_counts:
    see("counting on a middle class: the result", lambda: sc.project())
see("counting on a middle class: counts", lambda: list(seen_counts.values()))
see("counting on a middle class: a read after it", lambda: sc.validate)
see("counting on a middle class: called after it", lambda: sc.project())
see("counting on a middle class: what it leaves",
    lambda: ["validate" in vars(ProjectionC), "_counted_orig_validate" in vars(ProjectionC)])
with counting(ProjectionC, ("validate",)) as seen_counts:
    sc.project()
see("a second counting on that class: counts", lambda: list(seen_counts.values()))
see("a second counting on that class: a read after it", lambda: sc.validate)
with counting(StorageC, ("validate",)) as seen_counts:
    del ProjectionC.validate  # the restored copy goes, so that the lookup reaches the class that defines the kernel
    sc.project()
see("counting on the class that defines it: counts", lambda: list(seen_counts.values()))
see("counting on the class that defines it: a read after it", lambda: sc.validate)


# a chain under a decorated root: a middle class holds a fast-path descriptor for the inherited name, which is no new
# entry until it is replaced
class BaseChain:
    @ti.kernel
    def kd(self) -> ti.i32:
        return 61


@ti.data_oriented
class TopChain(BaseChain):
    pass


class MiddleChain(TopChain):
    pass


class LeafChain(MiddleChain):
    pass


lc = LeafChain()
see("a chain, before", lambda: lc.kd)
see("a chain, called", lambda: lc.kd())
saved_entry = inspect.getattr_static(MiddleChain, "kd")  # on: its descriptor; off: the kernel
MiddleChain.kd = lambda self: "on the middle"
see("a chain, a plain function over a middle class's descriptor", lambda: lc.kd())
see("a chain, the middle class's own read", lambda: MiddleChain().kd())
MiddleChain.kd = saved_entry
see("a chain, the entry put back", lambda: lc.kd)
see("a chain, the entry put back, called", lambda: lc.kd())
chain_patch = pytest.MonkeyPatch()
chain_patch.setattr(MiddleChain, "kd", lambda self: "patched")
see("a chain, monkeypatched on the middle class", lambda: lc.kd())
chain_patch.undo()
see("a chain, after the monkeypatch's undo", lambda: lc.kd)
TopChain.kd = lambda self: "on the root"
see("a chain, a plain function on the root, after the entry was put back on the middle class", lambda: lc.kd,
    DIFFERENCES)
del TopChain.kd
see("a chain, the root's entry deleted", lambda: lc.kd)
see("a chain, the root's own read after its entry was set and deleted", lambda: TopChain().kd, DIFFERENCES)


# a diamond, and a subclass of the descriptor's class
class BaseE:
    @ti.kernel
    def ke(self) -> ti.i32:
        return 71


class FirstE(BaseE):
    pass


class SecondE(BaseE):
    pass


@ti.data_oriented
class DiamondE(FirstE, SecondE):
    pass


de = DiamondE()
see("a diamond, before", lambda: de.ke)
SecondE.ke = lambda self: "on the second"
see("a diamond, set on its second side", lambda: de.ke())
FirstE.ke = lambda self: "on the first"
see("a diamond, set on its first side too", lambda: de.ke())
del FirstE.ke
see("a diamond, the first deleted", lambda: de.ke())
del SecondE.ke
see("a diamond, both deleted", lambda: de.ke)


class BaseH:
    @ti.kernel
    def kh(self) -> ti.i32:
        return 81


class MidH(BaseH):
    pass


@ti.data_oriented
class TopH(MidH):
    pass


class SubH(TopH):
    pass


sh = SubH()
see("a subclass of the holder, before", lambda: sh.kh)
MidH.kh = lambda self: "under the holder"
see("a subclass of the holder, set on a middle class", lambda: sh.kh())
del MidH.kh
see("a subclass of the holder, deleted", lambda: sh.kh)


# __bases__ assigned: listed
class BaseJ:
    @ti.kernel
    def kj(self) -> ti.i32:
        return 91


class OtherJ:
    @ti.kernel
    def kj(self) -> ti.i32:
        return 92

    @ti.kernel
    def new_name(self) -> ti.i32:
        return 93


class MidJ(BaseJ):
    pass


class NewMidJ(OtherJ):
    pass


@ti.data_oriented
class TopJ(MidJ):
    pass


rj = TopJ()
TopJ.__bases__ = (NewMidJ,)
see("__bases__ assigned, a kernel's name in both", lambda: rj.kj, DIFFERENCES)
see("__bases__ assigned, a kernel's name only the new base has", lambda: rj.new_name, DIFFERENCES)


# a staticmethod replaced on the first decorated class, under the name of a subclass's kernel
@ti.data_oriented
class TopR:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @staticmethod
    @ti.kernel
    def swap() -> ti.i32:
        return 1


@ti.data_oriented
class ChildR(TopR):
    @ti.kernel
    def swap(self) -> ti.i32:
        return 2


cr = ChildR()
see("a staticmethod on the first decorated class, under a subclass's kernel name", lambda: cr.swap)
TopR.swap = lambda self: "replaced"
see("that staticmethod replaced on the first decorated class", lambda: cr.swap, DIFFERENCES)


# what vars(Class) or getattr_static gives, used as a value
@ti.data_oriented
class TopA:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def fill(self):
        for i in self.data:
            self.data[i] = 5

    @staticmethod
    @ti.kernel
    def answer() -> ti.i32:
        return 6


ra = TopA()
ra.x = inspect.getattr_static(TopA, "fill")
see("an entry getattr_static gives, stored on an instance", lambda: ra.x)
see("an entry getattr_static gives, stored on an instance, called", lambda: after(ra.x, ra))
see("an entry getattr_static gives, in the instance __dict__", lambda: vars(ra)["x"])
ra.fill = inspect.getattr_static(TopA, "fill")
see("an entry getattr_static gives, stored under its own name", lambda: ra.fill)
see("an entry getattr_static gives, under its own name in the instance __dict__", lambda: vars(ra)["fill"])
del ra.fill
ra.y = inspect.getattr_static(Top, "mixin_prop")
see("a property getattr_static gives, stored on an instance", lambda: ra.y)
ra.z = inspect.getattr_static(TopA, "answer")
see("a staticmethod getattr_static gives, stored on an instance", lambda: ra.z)
TopA.alias = TopA.__dict__["fill"]
rb = TopA()
see("an entry copied under another name", lambda: rb.alias)
see("an entry copied under another name, called", lambda: after(rb.alias, rb))
see("an entry copied under another name, the class read", lambda: TopA.alias)
rb.alias = 3
see("an entry copied under another name, after a set",
    lambda: [rb.alias, describe(rb.fill)[0], sorted(vars(rb))], DIFFERENCES)
rc = TopA()
vars(rc)["held"] = inspect.getattr_static(TopA, "fill")
see("a descriptor put into vars() under a new name", lambda: rc.held, DIFFERENCES)
vars(rc)["fill"] = inspect.getattr_static(TopA, "fill")
see("a descriptor put into vars() under a kernel's name", lambda: rc.fill, DIFFERENCES)
on_the_hook = Logging()  # Logging's TopG went back to the 1.7.4 hook
on_the_hook.x = inspect.getattr_static(TopA, "fill")
see("a descriptor stored on an instance of a hierarchy on the hook", lambda: on_the_hook.x, DIFFERENCES)


@ti.data_oriented
class TopA2:
    @ti.kernel
    def fill(self):
        pass


TopA2.alias = TopA2.__dict__["fill"]
TopA2.fill = lambda self: "replaced"
see("an entry copied under another name, after the name's entry is replaced", lambda: TopA2().alias, DIFFERENCES)


class MixinA3:
    @ti.kernel
    def mk(self) -> ti.i32:
        return 14


@ti.data_oriented
class TopA3(MixinA3):
    pass


TopA3.alias = inspect.getattr_static(TopA3, "mk")  # on: an inherited kernel's descriptor, which replaced no entry


class LoggingA3(TopA3):  # hands TopA3's hierarchy back to the hook, which reads the copy as the kernel
    def __getattribute__(self, item):
        return super().__getattribute__(item)


see("an inherited entry copied under another name, after its hierarchy went back to the hook", lambda: TopA3().alias)
see("that entry, called", lambda: TopA3().alias())
see("attribute reads of a name with no entry", lambda: inspect.getattr_static(TopI, "stored").__name__, DIFFERENCES)


# a decorated subclass's property without a getter: __init_subclass__ put a descriptor over it first, and the
# decoration still finds it and hands the hierarchy back
@ti.data_oriented
class TopW:
    @ti.kernel
    def k(self) -> ti.i32:
        return 15


@ti.data_oriented
class ChildW(TopW):
    only = property(fset=lambda self, value: None)


see("a decorated subclass's property without a getter", lambda: ChildW().only)


# tools that read the class __dict__
@ti.data_oriented
class TopSpec:
    def __init__(self):
        self.n = 1

    @ti.kernel
    def own(self, a: ti.i32):
        pass


class SubSpec(TopSpec):
    pass


class SubSpec2(TopSpec):
    pass


def result_type(call):
    return type(call()).__name__


see("create_autospec(Class).kernel(1)", lambda: result_type(lambda: mock.create_autospec(TopSpec).own(1)), DIFFERENCES)
see("create_autospec(Class)().kernel(1)", lambda: result_type(lambda: mock.create_autospec(TopSpec)().own(1)),
    DIFFERENCES)
see("create_autospec(Class)().kernel(1, 2)", lambda: result_type(lambda: mock.create_autospec(TopSpec)().own(1, 2)),
    DIFFERENCES)


def patched_with_autospec():
    with mock.patch.object(TopSpec, "own", autospec=True) as patched:
        return type(patched).__name__


see("patch.object(Class, kernel, autospec=True)", patched_with_autospec, DIFFERENCES)
see("the kernel after patch.object", lambda: TopSpec().own)


def delete_inherited():
    del SubSpec.own


def monkeypatch_delete_inherited():
    pytest.MonkeyPatch().delattr(SubSpec2, "own")


see("del Sub.inherited", delete_inherited, DIFFERENCES)
see("the kernel after del Sub.inherited", lambda: SubSpec().own)
see("monkeypatch.delattr(Sub, inherited)", monkeypatch_delete_inherited, DIFFERENCES)
see("the kernel after monkeypatch.delattr(Sub, inherited)", lambda: SubSpec2().own)
see("deepcopy of what getattr_static gives",
    lambda: copy.deepcopy(inspect.getattr_static(TopDoc, "without_doc")).__name__, DIFFERENCES)
see("pickle of what getattr_static gives",
    lambda: type(pickle.dumps(inspect.getattr_static(TopDoc, "without_doc"))).__name__, DIFFERENCES)


# an attribute set on a copy of what getattr_static gives, under the name of one of its slots
@ti.data_oriented
class TopSet:
    @ti.kernel
    def k(self) -> ti.i32:
        return 16


def set_a_slot_name():
    entry = copy.copy(inspect.getattr_static(TopSet, "k"))
    entry.name = "x"
    state = getattr(entry, "state", ["no such attribute"])
    for slot in ("own", "raw", "state"):
        setattr(copy.copy(inspect.getattr_static(TopSet, "k")), slot, 1)
    return [entry.name, state[0]]


see("a slot's name set on a copy of what getattr_static gives", set_a_slot_name, DIFFERENCES)


# a name only the metaclass defines, stored on an instance
@ti.data_oriented
class TopMro:
    @ti.kernel
    def k(self) -> ti.i32:
        return 17


rm = TopMro()
rm.mro = TopMro.k
see("a kernel stored under the name mro", lambda: rm.mro)
see("Class.mro after an instance stored a kernel under that name", lambda: TopMro.mro()[0].__name__, DIFFERENCES)


# a kernel stored over a class entry that needs no descriptor, and a second decoration
@ti.data_oriented
class TopN:
    x = None
    y = 5

    @ti.kernel
    def k(self) -> ti.i32:
        return 18


rn = TopN()
rn.x = TopN.k
rn.y = TopN.k
see("a kernel stored over a None class entry", lambda: rn.x)
see("a kernel stored over a plain class entry", lambda: rn.y)
see("a sibling reads the None class entry", lambda: TopN().x)
ti.data_oriented(TopN)
see("a kernel stored over a None class entry, after a second decoration", lambda: rn.x, DIFFERENCES)
see("a kernel stored over a plain class entry, after a second decoration", lambda: rn.y, DIFFERENCES)


# an entry from vars() on an unrelated class, and in a staticmethod
class OtherClass:
    pass


OtherClass.k = TopMro.__dict__["k"]
see("a copy on an unrelated class, read", lambda: OtherClass().k, DIFFERENCES)
see("a copy on an unrelated class, read and called", lambda: OtherClass().k())
TopMro.s = staticmethod(TopMro.__dict__["k"])
see("a staticmethod of an entry from vars(), read", lambda: TopMro().s, DIFFERENCES)


# what a metaclass's __setattr__ sees of the decoration
META_LOG = []


class LoggingMeta(type):
    def __setattr__(cls, name, value):
        META_LOG.append(name)
        super().__setattr__(name, value)


@ti.data_oriented
class WithLoggingMeta(metaclass=LoggingMeta):
    @ti.kernel
    def k(self) -> ti.i32:
        return 19


see("what a metaclass's __setattr__ sees of the decoration", lambda: sorted(set(META_LOG)), DIFFERENCES)
see("a kernel of a class with such a metaclass", lambda: WithLoggingMeta().k)


# kernels under dunder names
@ti.data_oriented
class TopU:
    def __init__(self):
        self.data = ti.field(ti.i32, shape=2)

    @ti.kernel
    def __k__(self) -> ti.i32:
        return 7

    @ti.kernel
    def __call__(self) -> ti.i32:
        return 8

    @ti.kernel
    def plain(self) -> ti.i32:
        return 9


ru = TopU()
ru.__s__ = TopU.plain
see("a class's kernel under a dunder name", lambda: ru.__k__, DIFFERENCES)
see("the interpreter's call of a class's dunder kernel", lambda: ru(), DIFFERENCES)
see("a kernel stored on an instance under a dunder name", lambda: ru.__s__, DIFFERENCES)
see("TopU().plain()", lambda: TopU().plain())

DETAILS = details(Top, Child, Plain, Hooked, Deep, KwSub, Bypass, Diamond, WithMeta, Twice, WithClassMethod, TopP,
                  ChildP, TopG, TopZ, TopX, TopY, XY, KernelProp, TopKP, WriteOnly, Slotted, TopM, SubM,
                  SolverAB, TopS, SubS, TopG2, LoggingDirect, TopX2, TopY2, XY2, WithSlots, TopF, SubF, TopI,
                  TopT, SubT, ChildT, TopD, BypassD, TopU, TopDoc, SubDoc, BaseV1, DecoratedFirst,
                  BelowDecoratedFirst, BaseV2, UndecoratedFirst, BelowUndecoratedFirst, BaseV3, ReadsFirst, BaseV4,
                  HookedFirst, TopE, TopInt, TopR, ChildR, TopA, TopA2, TopA3, TopW, ChildW, TopSpec,
                  SubSpec, SubSpec2, TopSet, TopMro, TopN, WithLoggingMeta, TopB, SolverC, TopChain, MiddleChain,
                  LeafChain, DiamondE, TopH, SubH)
print("OBSERVED " + json.dumps(OBSERVED, sort_keys=True))
print("DIFFERENCES " + json.dumps(DIFFERENCES, sort_keys=True))
print("DETAILS " + json.dumps(DETAILS, sort_keys=True))
""".replace(
    "CPU_ARGUMENTS", CPU
)

DIFFERENCE_TAGS = [
    "a kernel put into vars()",
    "alias value",
    "class-level call",
    "own put back, class-level read",
    "super call",
    "super read of a shadowed name",
    "introspect TopM.mk while its base's entry is replaced",
    "a kernel set through object.__setattr__",
    "a call of the descriptor that getattr_static gives",
    "introspect TopI.stored, a name an instance stored a kernel under",
    "introspect ri.stored",
    "introspect ri.fill, an instance value over a kernel",
    "TopI's class __dict__ and dir",
    "classify_class_attrs",
    "a staticmethod kernel set later",
    "a staticmethod kernel set later, called with a scalar",
    "a staticmethod kernel set later, called without arguments",
    "a staticmethod kernel set later, called with an instance under the hook",
    "a classmethod kernel set later",
    "a classmethod kernel set later, called",
    "a Mock set later under a new name",
    "a Mock set later in place of the class's own kernel",
    "a property set later on an undecorated class, its getter returning a kernel",
    "a property set later on an undecorated class, its getter a kernel",
    "a property without a getter set later on a decorated class",
    "a property set later on a decorated class, its name defined before it",
    "a staticmethod set later on the first decorated class under a subclass's kernel name",
    "a subclass that skipped the hook, before its first set",
    "its property whose getter is a kernel, before the first set",
    "a class's kernel under a dunder name",
    "the interpreter's call of a class's dunder kernel",
    "a kernel stored on an instance under a dunder name",
    "what getattr_static gives: its type",
    "what getattr_static gives: isfunction, isroutine",
    "what getattr_static gives: repr",
    "what getattr_static gives: dir",
    "what getattr_static gives: the class read",
    "what getattr_static gives: a subclass's for an inherited kernel",
    "what getattr_static gives: an attribute set on it",
    "what getattr_static gives: __wrapped__",
    "what getattr_static gives: __get__",
    "what getattr_static gives: getfile",
    "what getattr_static gives: getdoc of a kernel that inherits its docstring",
    "a kernel stored on an instance before its class was decorated",
    "introspect TopM.gone after its base's entry is deleted",
    "that staticmethod replaced on the first decorated class",
    "an entry copied under another name, after a set",
    "a descriptor put into vars() under a new name",
    "a descriptor put into vars() under a kernel's name",
    "a descriptor stored on an instance of a hierarchy on the hook",
    "an entry copied under another name, after the name's entry is replaced",
    "attribute reads of a name with no entry",
    "create_autospec(Class).kernel(1)",
    "create_autospec(Class)().kernel(1)",
    "create_autospec(Class)().kernel(1, 2)",
    "patch.object(Class, kernel, autospec=True)",
    "del Sub.inherited",
    "monkeypatch.delattr(Sub, inherited)",
    "deepcopy of what getattr_static gives",
    "pickle of what getattr_static gives",
    "a slot's name set on a copy of what getattr_static gives",
    "Class.mro after an instance stored a kernel under that name",
    "a kernel stored over a None class entry, after a second decoration",
    "a kernel stored over a plain class entry, after a second decoration",
    "a copy on an unrelated class, read",
    "a staticmethod of an entry from vars(), read",
    "what a metaclass's __setattr__ sees of the decoration",
    "__bases__ assigned, a kernel's name in both",
    "__bases__ assigned, a kernel's name only the new base has",
    "a chain, a plain function on the root, after the entry was put back on the middle class",
    "a chain, the root's own read after its entry was set and deleted",
]


def plain(observation):
    """An observation of plain values (describe's "value" and "sequence") as Python values."""
    kind, payload = observation[0], observation[1]
    if kind == "sequence":
        return [plain(item) for item in payload]
    assert kind == "value", observation
    return payload


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
    """The scenarios with fast path d on (TI_LAUNCH_FASTPATH unset) and off (0, Taichi 1.7.4's code)."""
    return (
        run_script(tmp_path_factory.mktemp("on"), SCENARIOS, "unset"),
        run_script(tmp_path_factory.mktemp("off"), SCENARIOS, "0"),
    )


def test_every_read_set_delete_call_error_and_template_key_is_taichi_1_7_4s(runs):
    """Every observation outside FORK.md's differences (values of kernels, methods, fields, properties, cached and
    other descriptors, slots, plain, class and instance attributes of any name, a kernel or a Mock kept on an
    instance; sets and deletes and their errors; kernel results; template keys and their weak references; subclasses,
    decorated or not, with their own __init_subclass__ or one that skips fast path d's; a diamond, a metaclass, double
    decoration; each case handed back to the hook, read also through a __getattribute__ that skips the hook and under
    two hooks, and the class __dict__ of each; kernels of a base replaced, deleted and aliased after decoration, and
    the _is_staticmethod of a read after such a replacement; callable(), signature and unwrap of what
    inspect.getattr_static gives for kernels (own, inherited, staticmethod, classmethod), properties and other
    descriptors; ti.reset and ti.init) is the same with fast path d on as with Taichi 1.7.4's hook. The scenarios ran:
    the kernels moved the data, and calls that succeed in 1.7.4 (an alias, a class kernel kept on an instance, a
    module kernel read through an instance) give the same values. Each of these was a bug of an earlier draft:
    getattr_static's descriptor was not callable, so inspect.signature raised; the flag of a read after a replacement
    was the one at decoration; a hierarchy handed back to the hook kept the descriptors, which a __getattribute__ that
    skips the hook read; what getattr_static gives raised AttributeError for __name__, __globals__, _primal and the
    like; a base decorated after its decorated subclass left two hierarchies on the fast path, and a kernel read with
    the subclass's flag; an entry that getattr_static gave, stored on an instance, read back as the descriptor."""
    on, off = (tagged(done, "OBSERVED") for done in runs)
    assert sorted(on) == sorted(off)
    assert {tag: (on[tag], off[tag]) for tag in on if on[tag] != off[tag]} == {}
    assert plain(off["introspect Top.fill"])[0] is True and plain(off["introspect Top.fill"])[2] == "Top.fill"
    assert "value" in plain(off["introspect Top.fill"])[1]
    answer = plain(off["introspect Top.answer"])
    assert answer[0] is True and answer[1].startswith("() -> ") and answer[2] == "Top.answer"
    assert plain(off["introspect WithClassMethod.ck"]) == [False, ["TypeError", True], "WithClassMethod.ck"]
    assert plain(off["introspect Top.mixin_prop"])[:2] == [False, ["TypeError", True]]
    assert plain(off["a call of WithClassMethod.hook as getattr_static gives it"]) == [2, 2]
    assert off["a call of Top.answer as getattr_static gives it"][:2] == ["raises", "IndexError"]
    assert off["flip before"][3] is True and off["flip after the replacement"][3] is False
    assert off["flip() before"] == ["value", 1] and off["flip() after the replacement"] == ["value", 7]
    assert off["flip kept on an instance after the replacement, called"] == ["value", 8]
    assert off["later before"][3] is False and off["later after a staticmethod was set under it"][3] is True
    assert off["g2.fill through a __getattribute__ that skips the hook"][0] == "method"
    assert off["g2.fill() through a __getattribute__ that skips the hook"][:2] == ["raises", "AssertionError"]
    assert off["xy2.sx, a staticmethod kernel under two hooks"][3] is False
    # what getattr_static gives answers attribute reads as its entry; the entries read here are not all None
    reads = json.loads(plain(off["attribute reads of Top.fill"]))
    assert reads["__name__"] == "fill" and reads["__qualname__"] == "Top.fill"
    assert reads["__globals__"] == "taichi.lang.kernel_impl"
    assert reads["__code__"] == "wrapped" and reads["_is_wrapped_kernel"] is True and reads["_primal"] == "Kernel"
    assert reads["no_such_attribute"][0] == "AttributeError" and "value" in reads["__annotations__"]
    documented = json.loads(plain(off["attribute reads of TopDoc.with_doc"]))
    assert documented["__doc__"] == documented["getdoc"] == "A kernel's docstring."
    reads = json.loads(plain(off["attribute reads of Top.answer"]))
    assert reads["__func__"] == "function" and reads["__name__"] == "answer"
    assert reads["_is_wrapped_kernel"][0] == "AttributeError"  # a staticmethod object does not carry it on 3.10
    reads = json.loads(plain(off["attribute reads of Top.mixin_prop"]))
    assert reads["fget"] == "function" and reads["__doc__"] is None and reads["__name__"][0] == "AttributeError"
    assert json.loads(plain(off["attribute reads of WithSlots.held"]))["__objclass__"] == "type"
    assert json.loads(plain(off["attribute reads of TopE.args"]))["__objclass__"] == "type"
    assert plain(off["a copy of what getattr_static gives: callable, its __name__"]) == [True, "with_doc"]
    assert plain(off["a descriptor whose slots are not set"]) == [False, False]
    # an entry stored on an instance is read as the entry
    assert off["an entry getattr_static gives, stored on an instance"][0] == "bound kernel"
    assert plain(off["an entry getattr_static gives, stored on an instance, called"]) == [5, 5]
    assert off["an entry getattr_static gives, in the instance __dict__"][0] == "function"
    assert off["an entry getattr_static gives, stored under its own name"][0] == "bound kernel"
    assert off["a property getattr_static gives, stored on an instance"] == ["object", "property"]
    assert off["a staticmethod getattr_static gives, stored on an instance"] == ["object", "staticmethod"]
    assert off["an entry copied under another name"] == ["bound kernel", "fill", "TopA", False, "TopA.fill"]
    tag = "an inherited entry copied under another name, after its hierarchy went back to the hook"
    assert off[tag] == ["bound kernel", "mk", "TopA3", False, "MixinA3.mk"] and off["that entry, called"] == [
        "value",
        14,
    ]
    # a base decorated after a subclass: its hook, the innermost, bound each kernel with its own flag
    assert off["df.st_v, its base decorated later"] == [
        "bound kernel",
        "st_v",
        "DecoratedFirst",
        False,
        "DecoratedFirst.st_v",
    ]
    assert off["df.st_u, its base decorated later"][3] is True and off["df.st_u(), its base decorated later"][1] == 1
    assert off["df.st_v(), its base decorated later"][0] == "raises"
    assert off["df.kv(), its base decorated later"] == ["value", 4]
    assert off["uf.ku2(), its base decorated later"] == ["value", 7]
    assert off["uf.su2, its base decorated later"][3] is False and off["before.ku2()"] == ["value", 7]
    assert off["rf.kx, another __getattribute__ made before its base was decorated"][0] == "bound kernel"
    assert off["hf.s4, its subclass on the hook when the base was decorated"][3] is True
    assert off["hf.s5, its subclass on the hook when the base was decorated"][3] is False
    # a getset, a read-only one, a staticmethod on the first decorated class under a subclass's kernel name
    assert off["a getset read"] == ["sequence", [["value", 1], ["value", 2]]]
    assert plain(off["a getset after a set"]) == [[3], "3"]
    assert off["a getset deleted"][:2] == ["raises", "TypeError"]
    assert off["a getset set to what it refuses"][:2] == ["raises", "TypeError"]
    assert off["a read-only getset"] == ["value", 5] and off["a read-only getset set"][:2] == [
        "raises",
        "AttributeError",
    ]
    assert off["a staticmethod on the first decorated class, under a subclass's kernel name"][3] is True
    # the reads next to the listed differences
    assert off["the kernel after patch.object"] == ["bound kernel", "own", "TopSpec", False, "TopSpec.own"]
    assert off["the kernel after del Sub.inherited"] == ["bound kernel", "own", "SubSpec", False, "TopSpec.own"]
    assert off["the kernel after monkeypatch.delattr(Sub, inherited)"][0] == "bound kernel"
    assert off["a kernel stored under the name mro"] == ["bound kernel", "k", "TopMro", False, "TopMro.k"]
    assert off["a kernel stored over a None class entry"][0] == off["a kernel stored over a plain class entry"][0]
    assert off["a kernel stored over a None class entry"][0] == "bound kernel"
    assert off["a sibling reads the None class entry"] == ["value", None]
    assert off["a copy on an unrelated class, read and called"] == [
        "raises",
        "TaichiSyntaxError",
        "Please decorate class OtherClass with @ti.data_oriented",
    ]
    assert off["a kernel of a class with such a metaclass"][0] == "bound kernel"
    # an entry set later on a class between the descriptor's class and the entry's
    assert off["a kernel name set on an earlier class"] == ["method", "<lambda>", "SolverAB"]
    assert off["a kernel set on a middle class"] == ["bound kernel", "other", "TopB", False, "BaseB.other"]
    assert off["a kernel set on a middle class, called"] == ["value", 32]
    assert off["a plain function set on a middle class"][0] == "method"
    assert off["a plain function set on a middle class, called"] == ["value", "plain"]
    assert off["None set on a middle class"] == ["value", None] and off[
        "None set on a middle class, the class read"
    ] == [
        "value",
        None,
    ]
    assert off["a property set on a middle class, over an instance value"] == ["value", "property"]
    tag = "a decorated subclass's property without a getter"
    assert off[tag] == ["raises", "TypeError", "'NoneType' object is not callable"]
    assert off["the new entry deleted, the old kernel"] == ["bound kernel", "kb", "TopB", False, "BaseB.kb"]
    assert off["the new entry deleted, called"] == ["value", 31]
    assert off["counting on a middle class: counts"] == ["sequence", [["value", 1]]]
    assert off["counting on a middle class: the result"] == ["value", 51]
    assert off["counting on a middle class: a read after it"][0] == "bound kernel"
    assert off["counting on a middle class: called after it"] == ["value", 51]
    assert plain(off["counting on a middle class: what it leaves"]) == [True, False]
    assert off["a second counting on that class: counts"] == ["sequence", [["value", 1]]]
    assert off["counting on the class that defines it: counts"] == ["sequence", [["value", 1]]]
    assert off["a chain, before"] == ["bound kernel", "kd", "LeafChain", False, "BaseChain.kd"]
    assert off["a chain, a plain function over a middle class's descriptor"] == ["value", "on the middle"]
    assert off["a chain, the entry put back"] == ["bound kernel", "kd", "LeafChain", False, "BaseChain.kd"]
    assert off["a chain, monkeypatched on the middle class"] == ["value", "patched"]
    assert off["a chain, after the monkeypatch's undo"] == ["bound kernel", "kd", "LeafChain", False, "BaseChain.kd"]
    assert off["a diamond, set on its second side"] == ["value", "on the second"]
    assert off["a diamond, set on its first side too"] == ["value", "on the first"]
    assert off["a diamond, the first deleted"] == ["value", "on the second"]
    assert off["a diamond, both deleted"] == ["bound kernel", "ke", "DiamondE", False, "BaseE.ke"]
    assert off["a subclass of the holder, set on a middle class"] == ["value", "under the holder"]
    assert off["a subclass of the holder, deleted"][0] == "bound kernel"
    assert off["a kernel kept in a slot"][0] == "bound kernel"
    assert plain(off["a kernel kept in a slot, called"]) == [4, 4]
    assert off["a slot deleted"][:2] == ["raises", "AttributeError"]
    assert off["introspect TopI.method after an instance stored a kernel under it"][1][0] == ["value", True]
    assert off["ri.method, a kernel stored under the name of a method"][0] == "bound kernel"
    assert off["a staticmethod kernel set later, called with an instance on the fast path"] == ["value", 5]
    assert off["a Mock set later in place of a base's kernel, read through the subclass's descriptor"][:2] == [
        "raises",
        "AssertionError",
    ]
    assert off["a subclass that skipped the hook, after its first set"][0] == "bound kernel"
    assert off["its property whose getter is a kernel, after the first set"][:2] == ["raises", "AssertionError"]
    assert off["TopU().plain()"] == ["value", 9]
    assert off["r.fill(2)"] == ["sequence", [["value", 2], ["value", 4], ["value", 6], ["value", 8]]]
    assert off["r.use(r2)"] == ["sequence", [["value", 2], ["value", 3], ["value", 4], ["value", 5]]]
    alive = ["sequence", [["value", "weakref"], ["value", True]]]
    dead = ["sequence", [["value", "weakref"], ["value", False]]]
    assert off["use keys"] == ["sequence", [["sequence", [alive, alive]]]]  # self and the argument, weakly
    assert off["use keys after del"] == ["sequence", [["sequence", [alive, dead]]]]  # the key keeps no object alive
    assert off["r.fill"][0] == "bound kernel" and off["r.mixin_fill"][0] == "bound kernel"
    assert off["r.mixin_kernel_by_property"][0] == "bound kernel"
    assert off["class kernel under another name"][0] == "bound kernel"
    assert off["class kernel under another name, called"] == ["sequence", [["value", 3], ["value", 6]]]
    assert off["module kernel, called"] == ["sequence", [["value", 2], ["value", 2]]]
    assert off["module kernel flag"] == ["value", False] and off["mock value"][:2] == ["raises", "AssertionError"]
    assert off["m.alias(5)"] == ["sequence", [["value", 5], ["value", 5]]]
    assert off["m.mk replaced"][0] == "method" and off["m.mk(4) restored"][1][0] == ["value", 4]
    assert off["m.gone deleted"][:2] == ["raises", "AttributeError"]
    assert off["b.bypass_fill()"] == ["sequence", [["value", 13]] * 3]
    assert off["after reset r5.fill(4)"] == ["sequence", [["value", 4], ["value", 8], ["value", 12]]]


def test_the_differences_fork_md_lists(runs):
    """Each difference FORK.md change 7 lists, observed in both modes: off is 1.7.4's hook, on is fast path d. The
    class __dict__ holds descriptors: getattr_static gives one for a name an instance stored a kernel under and over
    an instance value, it stands for the old entry after a base's entry is replaced, dir and classify_class_attrs see
    it. An entry set on a class after it was made is read by Python's lookup: a kernel as a plain bound method (and a
    call gives the same values), a staticmethod or classmethod of a kernel as Python gives it (its call raises unless
    an instance on the fast path comes first), a Mock as the value, an undecorated class's property returning a
    kernel unbound and one whose getter is a kernel run, a decorated class's property without a getter or behind an
    earlier name as Python reads it, a staticmethod on the first decorated class without changing the flag of a
    subclass's kernel. A kernel name set on a class between the descriptor's class and the class that defined the
    kernel is not seen; super(), class-level calls and calls of the descriptor run the kernel where 1.7.4 raised;
    super() reads a shadowing instance attribute first; a kernel put into vars() or set through object.__setattr__ is
    not bound; dunder names get no descriptor; a subclass that skipped fast path d's __init_subclass__ reads its own
    entries as entries set later until its first set. What getattr_static gives is not the entry where only the type
    can answer (its type, isfunction, isroutine, getfile, a docstring inherited through getdoc, repr, dir, identity,
    equality, sets of attributes, __wrapped__, __get__). An entry copied onto a class under another name keeps its old
    name; a descriptor put into vars(), or stored on an instance of a hierarchy on the hook, stays a descriptor.
    create_autospec keeps self; patch.object with autospec gives a MagicMock; del and
    monkeypatch.delattr of an inherited name succeed; deepcopy and pickle of an entry fail; a set under a slot's name
    succeeds and leaves the state; Class.mro after an instance stored a kernel under it raises; a second decoration
    takes out a stored name over a None or plain entry; a copy on an unrelated class raises at the read; a staticmethod
    of an entry reads as the descriptor; a metaclass's __setattr__ does not see the descriptors. Nothing else differs
    (the other test's observations are all equal)."""
    on, off = (tagged(done, "DIFFERENCES") for done in runs)
    assert sorted(on) == sorted(off) == sorted(DIFFERENCE_TAGS)
    assert off["what getattr_static gives: its type"] == ["value", "function"]
    assert on["what getattr_static gives: its type"] == ["value", "_CallableDataOrientedKernel"]
    assert plain(off["what getattr_static gives: isfunction, isroutine"]) == [True, True]
    assert plain(on["what getattr_static gives: isfunction, isroutine"]) == [False, False]
    assert off["what getattr_static gives: repr"] == ["value", "<function"]
    assert on["what getattr_static gives: repr"] == ["value", "<_CallableDataOrientedKernel"]
    assert off["what getattr_static gives: dir"] == ["value", True] and on["what getattr_static gives: dir"] == [
        "value",
        False,
    ]
    assert plain(off["what getattr_static gives: the class read"]) == [True, True]
    assert plain(on["what getattr_static gives: the class read"]) == [False, False]
    tag = "what getattr_static gives: a subclass's for an inherited kernel"
    assert off[tag] == ["value", True] and on[tag] == ["value", False]
    tag = "what getattr_static gives: an attribute set on it"
    assert off[tag] == ["value", None] and on[tag][:2] == ["raises", "AttributeError"]
    tag = "what getattr_static gives: __wrapped__"
    assert off[tag] == ["value", False] and on[tag] == ["value", True]
    tag = "what getattr_static gives: __get__"
    assert off[tag] == ["method", "TopDoc.with_doc", "TopDoc"]
    assert on[tag] == ["bound kernel", "with_doc", "TopDoc", False, "TopDoc.with_doc"]
    tag = "what getattr_static gives: getfile"
    assert off[tag] == ["value", True] and on[tag][:2] == ["raises", "TypeError"]
    tag = "what getattr_static gives: getdoc of a kernel that inherits its docstring"
    assert off[tag] == ["value", "A kernel's docstring."] and on[tag] == ["value", None]
    tag = "a kernel stored on an instance before its class was decorated"
    assert off[tag][0] == "bound kernel" and on[tag] == ["function", "BaseV2.kb", True]
    tag = "introspect TopM.gone after its base's entry is deleted"
    assert off[tag][:2] == ["raises", "AttributeError"] and plain(on[tag])[0::2] == [True, "MixinM.gone"]
    tag = "that staticmethod replaced on the first decorated class"
    assert off[tag][3] is False and on[tag][3] is True
    tag = "an entry copied under another name, after a set"
    assert plain(off[tag]) == [3, "bound kernel", ["alias", "data"]]
    assert plain(on[tag]) == [3, "value", ["data", "fill"]]
    for tag in ("a descriptor put into vars() under a new name", "a descriptor put into vars() under a kernel's name"):
        assert off[tag][0] == "bound kernel" and on[tag] == ["object", "_CallableDataOrientedKernel"]
    tag = "a descriptor stored on an instance of a hierarchy on the hook"
    assert off[tag][0] == "bound kernel" and on[tag][:2] == ["raises", "AttributeError"]
    tag = "an entry copied under another name, after the name's entry is replaced"
    assert off[tag] == ["bound kernel", "fill", "TopA2", False, "TopA2.fill"]
    assert on[tag] == ["method", "<lambda>", "TopA2"]
    tag = "attribute reads of a name with no entry"
    assert off[tag][:2] == ["raises", "AttributeError"] and on[tag][:2] == ["raises", "AttributeError"]
    for tag in ("create_autospec(Class).kernel(1)", "create_autospec(Class)().kernel(1)"):
        assert off[tag] == ["value", "MagicMock"] and on[tag][:2] == ["raises", "TypeError"]
    tag = "create_autospec(Class)().kernel(1, 2)"
    assert off[tag][:2] == ["raises", "TypeError"] and on[tag] == ["value", "MagicMock"]
    tag = "patch.object(Class, kernel, autospec=True)"
    assert off[tag] == ["value", "function"] and on[tag] == ["value", "MagicMock"]
    for tag in ("del Sub.inherited", "monkeypatch.delattr(Sub, inherited)"):
        assert off[tag][:2] == ["raises", "AttributeError"] and on[tag] == ["value", None]
    tag = "deepcopy of what getattr_static gives"
    assert off[tag] == ["value", "without_doc"] and on[tag][:2] == ["raises", "TypeError"]
    tag = "pickle of what getattr_static gives"
    assert off[tag] == ["value", "bytes"] and on[tag][:2] == ["raises", "PicklingError"]
    tag = "a slot's name set on a copy of what getattr_static gives"
    assert plain(off[tag]) == ["x", "no such attribute"] and plain(on[tag]) == ["x", "k"]
    tag = "Class.mro after an instance stored a kernel under that name"
    assert off[tag] == ["value", "TopMro"] and on[tag][:2] == ["raises", "AttributeError"]
    for tag in (
        "a kernel stored over a None class entry, after a second decoration",
        "a kernel stored over a plain class entry, after a second decoration",
    ):
        assert off[tag] == ["bound kernel", "k", "TopN", False, "TopN.k"] and on[tag] == ["function", "TopN.k", True]
    tag = "a copy on an unrelated class, read"
    assert off[tag] == ["method", "TopMro.k", "OtherClass"]
    assert on[tag] == ["raises", "TaichiSyntaxError", "Please decorate class OtherClass with @ti.data_oriented"]
    tag = "a staticmethod of an entry from vars(), read"
    assert off[tag][0] == "bound kernel" and on[tag] == ["object", "_CallableDataOrientedKernel"]
    tag = "what a metaclass's __setattr__ sees of the decoration"
    assert plain(off[tag]) == ["__getattribute__", "_data_oriented"] and plain(on[tag]) == ["_data_oriented"]
    tag = "a chain, a plain function on the root, after the entry was put back on the middle class"
    assert off[tag] == ["bound kernel", "kd", "LeafChain", False, "BaseChain.kd"]
    assert on[tag] == ["method", "<lambda>", "LeafChain"]
    tag = "a chain, the root's own read after its entry was set and deleted"
    assert off[tag] == ["bound kernel", "kd", "TopChain", False, "BaseChain.kd"]
    assert on[tag] == ["method", "BaseChain.kd", "TopChain"]
    tag = "__bases__ assigned, a kernel's name in both"
    assert off[tag] == ["bound kernel", "kj", "TopJ", False, "OtherJ.kj"]
    assert on[tag] == ["bound kernel", "kj", "TopJ", False, "BaseJ.kj"]
    tag = "__bases__ assigned, a kernel's name only the new base has"
    assert off[tag] == ["bound kernel", "new_name", "TopJ", False, "OtherJ.new_name"]
    assert on[tag] == ["method", "OtherJ.new_name", "TopJ"]
    old, new = plain(off["introspect TopM.mk while its base's entry is replaced"]), plain(
        on["introspect TopM.mk while its base's entry is replaced"]
    )
    assert old[0] is new[0] is True and (old[2], new[2]) == ("<lambda>", "MixinM.mk") and "value" in new[1]
    assert off["a kernel set through object.__setattr__"][0] == "bound kernel"
    assert on["a kernel set through object.__setattr__"] == ["function", "Top.fill", True]
    assert off["a call of the descriptor that getattr_static gives"][:2] == ["raises", "AssertionError"]
    assert plain(on["a call of the descriptor that getattr_static gives"]) == [8, 8]
    tag = "introspect TopI.stored, a name an instance stored a kernel under"
    assert off[tag][:2] == ["raises", "AttributeError"]
    assert plain(on[tag]) == [False, ["TypeError", True], "_DataOrientedKernel"]
    assert plain(off["introspect ri.stored"])[0::2] == [True, "TopI.fill"]
    assert plain(on["introspect ri.stored"]) == [False, ["TypeError", True], "_DataOrientedKernel"]
    assert plain(off["introspect ri.fill, an instance value over a kernel"]) == [False, ["TypeError", True], "int"]
    assert plain(on["introspect ri.fill, an instance value over a kernel"])[0::2] == [True, "TopI.fill"]
    keys_off, kind_off, data_off, dir_off = plain(off["TopI's class __dict__ and dir"])
    keys_on, kind_on, data_on, dir_on = plain(on["TopI's class __dict__ and dir"])
    assert set(keys_on) - set(keys_off) == {"__init_subclass__", "__setattr__", "stored"}
    assert set(keys_off) - set(keys_on) == {"__getattribute__"}
    assert (kind_off, kind_on, data_off, data_on) == ("function", "_CallableDataOrientedKernel", False, True)
    assert set(dir_on) - set(dir_off) == {"stored"} and set(dir_off) <= set(dir_on)
    assert plain(off["classify_class_attrs"]) == [
        [
            ["answer", "static method", "Top"],
            ["fill", "method", "Top"],
            ["method", "method", "Top"],
            ["mixin_fill", "method", "Mixin"],
            ["mixin_prop", "property", "Mixin"],
        ],
        [["ck", "class method", "WithClassMethod"]],
    ]
    assert plain(on["classify_class_attrs"]) == [
        [
            ["answer", "method", "Top"],
            ["fill", "method", "Top"],
            ["method", "method", "Top"],
            ["mixin_fill", "method", "Top"],
            ["mixin_prop", "data", "Top"],
        ],
        [["ck", "method", "WithClassMethod"]],
    ]
    assert off["a staticmethod kernel set later"] == ["bound kernel", "twice", "SubT", True, "TopT.twice"]
    assert on["a staticmethod kernel set later"] == ["function", "TopT.twice", True]
    assert off["a staticmethod kernel set later, called with a scalar"] == ["value", 6]
    assert on["a staticmethod kernel set later, called with a scalar"] == [
        "raises",
        "TaichiSyntaxError",
        "Please decorate class int with @ti.data_oriented",
    ]
    assert off["a staticmethod kernel set later, called without arguments"][0] == "raises"
    assert on["a staticmethod kernel set later, called without arguments"][:2] == ["raises", "IndexError"]
    assert off["a staticmethod kernel set later, called with an instance under the hook"] == ["value", 0]
    assert on["a staticmethod kernel set later, called with an instance under the hook"][:2] == [
        "raises",
        "AssertionError",
    ]
    assert off["a classmethod kernel set later"] == ["bound kernel", "plain", "SubT", False, "TopT.plain"]
    assert on["a classmethod kernel set later"] == ["method", "TopT.plain", "type"]
    assert off["a classmethod kernel set later, called"] == ["value", 6]
    assert on["a classmethod kernel set later, called"] == [
        "raises",
        "TaichiSyntaxError",
        "Please decorate class type with @ti.data_oriented",
    ]
    for tag in ("a Mock set later under a new name", "a Mock set later in place of the class's own kernel"):
        assert off[tag][:2] == ["raises", "AssertionError"] and on[tag] == ["object", "Mock"]
    tag = "a property set later on an undecorated class, its getter returning a kernel"
    assert off[tag] == ["bound kernel", "mixin_peek", "SubT", False, "MixinT.mixin_peek"]
    assert on[tag] == ["function", "MixinT.mixin_peek", True]
    tag = "a property set later on an undecorated class, its getter a kernel"
    assert off[tag][:2] == ["raises", "AssertionError"] and on[tag] == ["value", 105]
    tag = "a property without a getter set later on a decorated class"
    assert off[tag] == ["raises", "TypeError", "'NoneType' object is not callable"]
    assert on[tag][:2] == ["raises", "AttributeError"]
    tag = "a property set later on a decorated class, its name defined before it"
    assert off[tag] == ["value", "property"] and on[tag] == ["value", 1]
    tag = "a staticmethod set later on the first decorated class under a subclass's kernel name"
    assert off[tag] == ["bound kernel", "own_k", "ChildT", True, "ChildT.own_k"]
    assert on[tag] == ["bound kernel", "own_k", "ChildT", False, "ChildT.own_k"]
    tag = "a subclass that skipped the hook, before its first set"
    assert off[tag] == ["bound kernel", "bypass_d", "BypassD", False, "BypassD.bypass_d"]
    assert on[tag] == ["method", "BypassD.bypass_d", "BypassD"]
    tag = "its property whose getter is a kernel, before the first set"
    assert off[tag][:2] == ["raises", "AssertionError"] and on[tag] == ["value", 3]
    assert off["a class's kernel under a dunder name"] == ["bound kernel", "__k__", "TopU", False, "TopU.__k__"]
    assert on["a class's kernel under a dunder name"] == ["method", "TopU.__k__", "TopU"]
    assert off["the interpreter's call of a class's dunder kernel"][:2] == ["raises", "AssertionError"]
    assert on["the interpreter's call of a class's dunder kernel"] == ["value", 8]
    tag = "a kernel stored on an instance under a dunder name"
    assert off[tag] == ["bound kernel", "plain", "TopU", False, "TopU.plain"]
    assert on[tag] == ["function", "TopU.plain", True]
    assert off["alias value"][0] == "bound kernel" and on["alias value"][0] == "method"
    assert off["own put back, class-level read"][0] == "bound kernel"
    assert on["own put back, class-level read"][0] == "method"
    assert off["super call"][:2] == ["raises", "AssertionError"]
    assert on["super call"] == ["sequence", [["value", 8], ["value", 8]]]
    assert off["super read of a shadowed name"][0] == "method" and on["super read of a shadowed name"] == ["value", 5]
    assert off["class-level call"][:2] == ["raises", "AssertionError"]
    assert on["class-level call"] == ["sequence", [["value", 8], ["value", 8]]]
    assert off["a kernel put into vars()"][0] == "bound kernel" and on["a kernel put into vars()"][0] == "function"


def test_descriptors_where_they_keep_the_hooks_reads_and_the_hook_back_for_each_case_they_cannot(runs):
    """On: a decorated class and its subclasses get descriptors for their kernel names (inherited ones included) and
    no hook; each case FORK.md names gets the hook back for its whole decorated hierarchy: a property whose getter is
    a kernel (in the class, KernelProp, or in an undecorated base, TopKP), a decorated class's property without a
    getter (WriteOnly, and ChildW with its base TopW: __init_subclass__ put a descriptor over it first) or defined
    again in a subclass (TopP and its decorated ChildP), another __getattribute__
    (TopG, TopG2) or __setattr__ (TopZ), no instance __dict__ (Slotted), two decorated hierarchies in one MRO (TopX
    and TopY, TopX2 and TopY2), a kernel-getter property of a subclass that skipped fast path d's __init_subclass__
    (TopD, at its instance's first set), a class decorated after a decorated subclass (BaseV1 and DecoratedFirst, and
    BaseV4 with HookedFirst, already on the hook), a subclass with another __getattribute__ made before its base was
    decorated (BaseV3); and such a hierarchy keeps no descriptor. An undecorated subclass made before its base was
    decorated gets descriptors at the decoration. A slot and a getset get a descriptor, a dunder name none. Off: every
    decorated class has the hook and no descriptor."""
    on, off = (tagged(done, "DETAILS") for done in runs)
    undecorated = ("Plain", "Hooked", "Deep", "KwSub", "Bypass", "XY", "SubM", "SubS", "LoggingDirect", "XY2", "SubF")
    undecorated += ("SubT", "BypassD", "SubDoc", "BelowDecoratedFirst", "UndecoratedFirst", "BelowUndecoratedFirst")
    undecorated += ("ReadsFirst", "SubSpec", "SubSpec2", "MiddleChain", "LeafChain", "SubH")
    assert all(off[name] == dict(hook=True, fast=False, descriptors=[]) for name in off if name not in undecorated)
    # Top's kernels, the properties and the cached_property of its undecorated base (the hook tested their values),
    # and the names an instance stored a kernel or a Mock under; not the one put into vars().
    expected = ["answer", "cached", "fill", "first", "mixin_fill", "mixin_kernel_by_property", "mixin_prop", "use"]
    expected += ["layer_recorded", "mixin_settable", "recorded"]  # other descriptors, and an undecorated setter
    expected += ["hook", "mocked", "other"]
    assert on["Top"] == dict(hook=False, fast=True, descriptors=sorted(expected))
    assert on["Child"]["fast"] and {"child_fill", "fill", "mixin_fill"} <= set(on["Child"]["descriptors"])
    for name, own in (("Plain", "plain_fill"), ("Deep", "deep_fill"), ("KwSub", "kw_fill"), ("Bypass", "bypass_fill")):
        assert on[name]["hook"] is False and on[name]["fast"] is False and own in on[name]["descriptors"]
    for fast in ("Diamond", "WithMeta", "TwiceBody", "WithClassMethod", "TopM", "SolverAB", "TopS"):
        assert on[fast]["hook"] is False and on[fast]["fast"] is True
    for reverted in ("TopP", "ChildP", "TopG", "TopZ", "TopX", "TopY", "TopG2", "TopX2", "TopY2", "TopD"):
        assert on[reverted] == dict(hook=True, fast=False, descriptors=[])
    for undecorated_reverted in ("XY", "XY2", "BypassD"):
        assert on[undecorated_reverted] == dict(hook=False, fast=False, descriptors=[])
    for refused in ("KernelProp", "TopKP", "WriteOnly", "Slotted", "TopW", "ChildW"):
        assert on[refused] == dict(hook=True, fast=False, descriptors=[])
    assert on["WithSlots"] == dict(hook=False, fast=True, descriptors=["fill", "held"])
    assert on["TopU"] == dict(hook=False, fast=True, descriptors=["plain"])
    assert on["TopI"]["descriptors"] == ["answer", "fill", "method", "stored"]
    assert on["TopF"]["descriptors"] == ["plain"] and on["SubF"]["descriptors"] == ["flip", "later", "plain"]
    assert on["TopT"]["descriptors"] == ["mixin_peek", "peek", "twice"]  # TopT.plain became a Mock
    assert on["ChildT"]["descriptors"] == ["mixin_peek", "own_k", "peek", "plain", "twice"]
    for reverted in ("BaseV1", "DecoratedFirst", "BaseV3", "BaseV4", "HookedFirst"):
        assert on[reverted] == dict(hook=True, fast=False, descriptors=[])
    assert on["BelowDecoratedFirst"] == dict(hook=False, fast=False, descriptors=[])
    assert on["ReadsFirst"] == dict(hook=True, fast=False, descriptors=[])  # its own __getattribute__
    assert on["TopA3"] == dict(hook=True, fast=False, descriptors=[])
    # its descriptor of the inherited kernel deleted
    assert on["SubSpec"] == dict(hook=False, fast=False, descriptors=[])
    assert on["TopN"] == dict(hook=False, fast=True, descriptors=["k"])  # the second decoration took x and y out
    assert on["BaseV2"] == dict(hook=False, fast=True, descriptors=["kb"])
    assert on["UndecoratedFirst"] == dict(hook=False, fast=False, descriptors=["kb", "ku2", "su2"])
    assert on["BelowUndecoratedFirst"] == dict(hook=False, fast=False, descriptors=["kb", "kb2", "ku2", "su2"])
    assert on["TopE"]["fast"] and {"args", "fill"} == set(on["TopE"]["descriptors"])
    assert on["TopInt"]["fast"] and {"real", "imag", "numerator", "denominator"} == set(on["TopInt"]["descriptors"])
