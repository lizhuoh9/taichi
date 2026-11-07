# Changed in the fork lizhuoh9/taichi on 2026-11-06 (see FORK.md): launch fast paths in extract_arg, _process_args and
# Kernel.launch_kernel, on unless TI_LAUNCH_FASTPATH=0; they return and raise what Taichi 1.7.4 returns and raises.
# Changed on 2026-11-07: fast path d, descriptors for the kernels of a data_oriented class instead of its hook; it too
# returns and raises what 1.7.4 does, except the differences listed under change 7 of FORK.md.
import ast
import functools
import inspect
import operator
import re
import sys
import textwrap
import typing
import types
import warnings
import weakref

import numpy as np
import taichi.lang
from taichi._lib import core as _ti_core
from taichi.lang import impl, ops, runtime_ops
from taichi.lang.any_array import AnyArray
from taichi.lang._wrap_inspect import getsourcefile, getsourcelines
from taichi.lang.argpack import ArgPackType, ArgPack
from taichi.lang.ast import (
    ASTTransformerContext,
    KernelSimplicityASTChecker,
    transform_tree,
)
from taichi.lang.ast.ast_transformer_utils import ReturnStatus
from taichi.lang.enums import AutodiffMode, Layout
from taichi.lang.exception import (
    TaichiCompilationError,
    TaichiRuntimeError,
    TaichiRuntimeTypeError,
    TaichiSyntaxError,
    TaichiTypeError,
    handle_exception_from_cpp,
)
from taichi.lang.expr import Expr
from taichi.lang.kernel_arguments import KernelArgument
from taichi.lang.matrix import MatrixType
from taichi.lang.shell import _shell_pop_print
from taichi.lang.struct import StructType
from taichi.lang.util import cook_dtype, has_paddle, has_pytorch, to_taichi_type
from taichi.types import (
    ndarray_type,
    primitive_types,
    sparse_matrix_builder,
    template,
    texture_type,
)
from taichi.types.compound_types import CompoundType
from taichi.types.utils import is_signed

from taichi import _logging
from taichi._fork import LAUNCH_FASTPATH as _LAUNCH_FASTPATH

# The launch fast paths of the fork (FORK.md), each of which hands every case it does not cover to Taichi's code:
#   a. TaichiCallableTemplateMapper.extract_arg keys a template argument by a decision made once per type;
#   b. _process_args returns a call's positional arguments when they fill every parameter;
#   c. Kernel.launch_kernel launches a kernel whose non-template parameters are all primitive scalars with a plan made
#      at its first launch, and keeps each compiled kernel's CompiledKernelData for its program;
#   d. data_oriented gives a class a descriptor for each of its kernels instead of a __getattribute__ hook, so that
#      reading any other attribute of an instance is Python's own lookup (see data_oriented).
# _LAUNCH_FASTPATH is taichi._fork.LAUNCH_FASTPATH: TI_LAUNCH_FASTPATH=0 at import turns all four off.
_KEY_FROM_RELEASED_BRANCHES, _KEY_IS_ARGUMENT, _KEY_IS_WEAK_REFERENCE = 0, 1, 2
_TEMPLATE_KEY_KINDS = {}  # a. type -> one of the three kinds above
_PRIMITIVE_KEY_TYPES = (int, float, bool, str, type(None))
_EMPTY = inspect.Parameter.empty
_SCALAR_REAL, _SCALAR_SIGNED, _SCALAR_UNSIGNED = 0, 1, 2
_REAL_ARGUMENT_TYPES = (float, int, np.floating, np.integer)  # what launch_kernel accepts for a real parameter
_INTEGER_ARGUMENT_TYPES = (int, np.integer)  # and for an integer parameter
_MAX_ARG_NUM = 64  # launch_kernel's limit on argument slots
_UNPLANNED = object()


def func(fn, is_real_function=False):
    """Marks a function as callable in Taichi-scope.

    This decorator transforms a Python function into a Taichi one. Taichi
    will JIT compile it into native instructions.

    Args:
        fn (Callable): The Python function to be decorated
        is_real_function (bool): Whether the function is a real function

    Returns:
        Callable: The decorated function

    Example::

        >>> @ti.func
        >>> def foo(x):
        >>>     return x + 2
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     print(foo(40))  # 42
    """
    is_classfunc = _inside_class(level_of_class_stackframe=3 + is_real_function)

    fun = Func(fn, _classfunc=is_classfunc, is_real_function=is_real_function)

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return fun.__call__(*args, **kwargs)

    decorated._is_taichi_function = True
    decorated._is_real_function = is_real_function
    decorated.func = fun
    return decorated


def real_func(fn):
    return func(fn, is_real_function=True)


def pyfunc(fn):
    """Marks a function as callable in both Taichi and Python scopes.

    When called inside the Taichi scope, Taichi will JIT compile it into
    native instructions. Otherwise it will be invoked directly as a
    Python function.

    See also :func:`~taichi.lang.kernel_impl.func`.

    Args:
        fn (Callable): The Python function to be decorated

    Returns:
        Callable: The decorated function
    """
    is_classfunc = _inside_class(level_of_class_stackframe=3)
    fun = Func(fn, _classfunc=is_classfunc, _pyfunc=True)

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return fun.__call__(*args, **kwargs)

    decorated._is_taichi_function = True
    decorated._is_real_function = False
    decorated.func = fun
    return decorated


def _get_tree_and_ctx(
    self,
    excluded_parameters=(),
    is_kernel=True,
    arg_features=None,
    args=None,
    ast_builder=None,
    is_real_function=False,
):
    file = getsourcefile(self.func)
    src, start_lineno = getsourcelines(self.func)
    src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
    tree = ast.parse(textwrap.dedent("\n".join(src)))

    func_body = tree.body[0]
    func_body.decorator_list = []

    global_vars = _get_global_vars(self.func)

    if is_kernel or is_real_function:
        # inject template parameters into globals
        for i in self.template_slot_locations:
            template_var_name = self.arguments[i].name
            global_vars[template_var_name] = args[i]

    return tree, ASTTransformerContext(
        excluded_parameters=excluded_parameters,
        is_kernel=is_kernel,
        func=self,
        arg_features=arg_features,
        global_vars=global_vars,
        argument_data=args,
        src=src,
        start_lineno=start_lineno,
        file=file,
        ast_builder=ast_builder,
        is_real_function=is_real_function,
    )


def _process_args(self, args, kwargs):
    if _LAUNCH_FASTPATH and not kwargs and len(args) == len(self.arguments):
        # Fast path b: positional arguments only, one per parameter and none of them inspect.Parameter.empty; the
        # loops below would return the list of the arguments.  Every other call takes them, and their errors.
        for arg in args:
            if arg is _EMPTY:
                break
        else:
            return list(args)
    ret = [argument.default for argument in self.arguments]
    len_args = len(args)

    if len_args > len(ret):
        arg_str = ", ".join([str(arg) for arg in args])
        expected_str = ", ".join([f"{arg.name} : {arg.annotation}" for arg in self.arguments])
        msg = f"Too many arguments. Expected ({expected_str}), got ({arg_str})."
        raise TaichiSyntaxError(msg)

    for i, arg in enumerate(args):
        ret[i] = arg

    for key, value in kwargs.items():
        found = False
        for i, arg in enumerate(self.arguments):
            if key == arg.name:
                if i < len_args:
                    raise TaichiSyntaxError(f"Multiple values for argument '{key}'.")
                ret[i] = value
                found = True
                break
        if not found:
            raise TaichiSyntaxError(f"Unexpected argument '{key}'.")

    for i, arg in enumerate(ret):
        if arg is inspect.Parameter.empty:
            if self.arguments[i].annotation is inspect._empty:
                raise TaichiSyntaxError(f"Parameter `{self.arguments[i].name}` missing.")
            else:
                raise TaichiSyntaxError(
                    f"Parameter `{self.arguments[i].name} : {self.arguments[i].annotation}` missing."
                )

    return ret


class Func:
    function_counter = 0

    def __init__(self, _func, _classfunc=False, _pyfunc=False, is_real_function=False):
        self.func = _func
        self.func_id = Func.function_counter
        Func.function_counter += 1
        self.compiled = None
        self.classfunc = _classfunc
        self.pyfunc = _pyfunc
        self.is_real_function = is_real_function
        self.arguments = []
        self.return_type = None
        self.extract_arguments()
        self.template_slot_locations = []
        for i, arg in enumerate(self.arguments):
            if isinstance(arg.annotation, template):
                self.template_slot_locations.append(i)
        self.mapper = TaichiCallableTemplateMapper(self.arguments, self.template_slot_locations)
        self.taichi_functions = {}  # The |Function| class in C++
        self.has_print = False

    def __call__(self, *args, **kwargs):
        args = _process_args(self, args, kwargs)

        if not impl.inside_kernel():
            if not self.pyfunc:
                raise TaichiSyntaxError("Taichi functions cannot be called from Python-scope.")
            return self.func(*args)

        if self.is_real_function:
            if impl.get_runtime().current_kernel.autodiff_mode != AutodiffMode.NONE:
                raise TaichiSyntaxError("Real function in gradient kernels unsupported.")
            instance_id, arg_features = self.mapper.lookup(args)
            key = _ti_core.FunctionKey(self.func.__name__, self.func_id, instance_id)
            if self.compiled is None:
                self.compiled = {}
            if key.instance_id not in self.compiled:
                self.do_compile(key=key, args=args, arg_features=arg_features)
            return self.func_call_rvalue(key=key, args=args)
        tree, ctx = _get_tree_and_ctx(
            self,
            is_kernel=False,
            args=args,
            ast_builder=impl.get_runtime().current_kernel.ast_builder(),
            is_real_function=self.is_real_function,
        )
        ret = transform_tree(tree, ctx)
        if not self.is_real_function:
            if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                raise TaichiSyntaxError("Function has a return type but does not have a return statement")
        return ret

    def func_call_rvalue(self, key, args):
        # Skip the template args, e.g., |self|
        assert self.is_real_function
        non_template_args = []
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        for i, kernel_arg in enumerate(self.arguments):
            anno = kernel_arg.annotation
            if not isinstance(anno, template):
                if id(anno) in primitive_types.type_ids:
                    non_template_args.append(ops.cast(args[i], anno))
                elif isinstance(anno, primitive_types.RefType):
                    non_template_args.append(_ti_core.make_reference(args[i].ptr, dbg_info))
                elif isinstance(anno, ndarray_type.NdarrayType):
                    if not isinstance(args[i], AnyArray):
                        raise TaichiTypeError(
                            f"Expected ndarray in the kernel argument for argument {kernel_arg.name}, got {args[i]}"
                        )
                    non_template_args += _ti_core.get_external_tensor_real_func_args(args[i].ptr, dbg_info)
                else:
                    non_template_args.append(args[i])
        non_template_args = impl.make_expr_group(non_template_args)
        func_call = (
            impl.get_runtime()
            .compiling_callable.ast_builder()
            .insert_func_call(self.taichi_functions[key.instance_id], non_template_args, dbg_info)
        )
        if self.return_type is None:
            return None
        func_call = Expr(func_call)
        ret = []

        for i, return_type in enumerate(self.return_type):
            if id(return_type) in primitive_types.type_ids:
                ret.append(
                    Expr(
                        _ti_core.make_get_element_expr(
                            func_call.ptr, (i,), _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
                        )
                    )
                )
            elif isinstance(return_type, (StructType, MatrixType)):
                ret.append(return_type.from_taichi_object(func_call, (i,)))
            else:
                raise TaichiTypeError(f"Unsupported return type for return value {i}: {return_type}")
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

    def do_compile(self, key, args, arg_features):
        tree, ctx = _get_tree_and_ctx(
            self, is_kernel=False, args=args, arg_features=arg_features, is_real_function=self.is_real_function
        )
        fn = impl.get_runtime().prog.create_function(key)

        def func_body():
            old_callable = impl.get_runtime().compiling_callable
            impl.get_runtime().compiling_callable = fn
            ctx.ast_builder = fn.ast_builder()
            transform_tree(tree, ctx)
            impl.get_runtime().compiling_callable = old_callable

        self.taichi_functions[key.instance_id] = fn
        self.compiled[key.instance_id] = func_body
        self.taichi_functions[key.instance_id].set_function_body(func_body)

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect.Signature.empty, None):
            self.return_type = sig.return_annotation
            if sys.version_info >= (3, 9):
                if (
                    isinstance(self.return_type, (types.GenericAlias, typing._GenericAlias))
                    and self.return_type.__origin__ is tuple
                ):
                    self.return_type = self.return_type.__args__
            else:
                if isinstance(self.return_type, typing._GenericAlias) and self.return_type.__origin__ is tuple:
                    self.return_type = self.return_type.__args__
            if not isinstance(self.return_type, (list, tuple)):
                self.return_type = (self.return_type,)
            for i, return_type in enumerate(self.return_type):
                if return_type is Ellipsis:
                    raise TaichiSyntaxError("Ellipsis is not supported in return type annotations")
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TaichiSyntaxError("Taichi functions do not support variable keyword parameters (i.e., **kwargs)")
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TaichiSyntaxError("Taichi functions do not support variable positional parameters (i.e., *args)")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise TaichiSyntaxError("Taichi functions do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TaichiSyntaxError('Taichi functions only support "positional or keyword" parameters')
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                if i == 0 and self.classfunc:
                    annotation = template()
                # TODO: pyfunc also need type annotation check when real function is enabled,
                #       but that has to happen at runtime when we know which scope it's called from.
                elif not self.pyfunc and self.is_real_function:
                    raise TaichiSyntaxError(
                        f"Taichi function `{self.func.__name__}` parameter `{arg_name}` must be type annotated"
                    )
            else:
                if isinstance(annotation, ndarray_type.NdarrayType):
                    pass
                elif isinstance(annotation, MatrixType):
                    pass
                elif isinstance(annotation, StructType):
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif isinstance(annotation, template):
                    pass
                elif isinstance(annotation, primitive_types.RefType):
                    pass
                else:
                    raise TaichiSyntaxError(f"Invalid type annotation (argument {i}) of Taichi function: {annotation}")
            self.arguments.append(KernelArgument(annotation, param.name, param.default))


def _template_key_kind(arg_type):
    """How the released branches of TaichiCallableTemplateMapper.extract_arg key every instance of arg_type as a
    template argument (fast path a).

    _KEY_IS_ARGUMENT: the instance itself, for int, float, bool, str and None, NumPy's scalar types other than the
    structured np.void and np.record (whose fields belong to each instance), and Taichi's field classes with plain
    attribute lookup.  _KEY_IS_WEAK_REFERENCE: a weak reference to it, for classes with a _data_oriented attribute of
    their own or of a base (what an instance finds; a metaclass's attribute is not).  _KEY_FROM_RELEASED_BRANCHES for
    the types of the released branches (SNode, Expr, tuple, Ndarray, list, dict, set) and every type not listed: those
    keys are made per instance.  The decision is made at a type's first call; a class changed afterwards keeps it.
    """
    # Imported here: the attribute taichi.lang.field is the function ti.field, not the module.
    from taichi.lang.field import Field  # pylint: disable=C0415

    own_branches = (taichi.lang.snode.SNode, Expr, _ti_core.Expr, tuple, taichi.lang._ndarray.Ndarray, list, dict, set)
    if issubclass(arg_type, own_branches):
        return _KEY_FROM_RELEASED_BRANCHES
    if any("_data_oriented" in vars(klass) for klass in arg_type.__mro__):
        return _KEY_IS_WEAK_REFERENCE
    if arg_type in _PRIMITIVE_KEY_TYPES:
        return _KEY_IS_ARGUMENT
    if (
        issubclass(arg_type, np.generic)
        and not issubclass(arg_type, np.void)
        and arg_type.__module__ == "numpy"
        and arg_type.__dictoffset__ == 0
    ):
        return _KEY_IS_ARGUMENT  # NumPy's scalar types: no instance attributes and no fields
    plain_lookup = arg_type.__getattribute__ is object.__getattribute__ and not hasattr(arg_type, "__getattr__")
    if issubclass(arg_type, Field) and plain_lookup:
        return _KEY_IS_ARGUMENT
    return _KEY_FROM_RELEASED_BRANCHES


class TaichiCallableTemplateMapper:
    def __init__(self, arguments, template_slot_locations):
        self.arguments = arguments
        self.num_args = len(arguments)
        self.template_slot_locations = template_slot_locations
        self.mapping = {}

    @staticmethod
    def extract_arg(arg, anno, arg_name):
        if isinstance(anno, template):
            if _LAUNCH_FASTPATH:
                # Fast path a: the key that the branches below give every instance of this type
                arg_type = type(arg)
                kind = _TEMPLATE_KEY_KINDS.get(arg_type)
                if kind is None:
                    kind = _TEMPLATE_KEY_KINDS[arg_type] = _template_key_kind(arg_type)
                if kind == _KEY_IS_ARGUMENT:
                    return arg
                if kind == _KEY_IS_WEAK_REFERENCE:
                    return weakref.ref(arg)
            if isinstance(arg, taichi.lang.snode.SNode):
                return arg.ptr
            if isinstance(arg, taichi.lang.expr.Expr):
                return arg.ptr.get_underlying_ptr_address()
            if isinstance(arg, _ti_core.Expr):
                return arg.get_underlying_ptr_address()
            if isinstance(arg, tuple):
                return tuple(TaichiCallableTemplateMapper.extract_arg(item, anno, arg_name) for item in arg)
            if isinstance(arg, taichi.lang._ndarray.Ndarray):
                raise TaichiRuntimeTypeError(
                    "Ndarray shouldn't be passed in via `ti.template()`, please annotate your kernel using `ti.types.ndarray(...)` instead"
                )

            if isinstance(arg, (list, tuple, dict, set)) or hasattr(arg, "_data_oriented"):
                # [Composite arguments] Return weak reference to the object
                # Taichi kernel will cache the extracted arguments, thus we can't simply return the original argument.
                # Instead, a weak reference to the original value is returned to avoid memory leak.

                # TODO(zhanlue): replacing "tuple(args)" with "hash of argument values"
                # This can resolve the following issues:
                # 1. Invalid weak-ref will leave a dead(dangling) entry in both caches: "self.mapping" and "self.compiled_functions"
                # 2. Different argument instances with same type and same value, will get templatized into seperate kernels.
                return weakref.ref(arg)

            # [Primitive arguments] Return the value
            return arg
        if isinstance(anno, ArgPackType):
            if not isinstance(arg, ArgPack):
                raise TaichiRuntimeTypeError(f"Argument {arg_name} must be a argument pack, got {type(arg)}")
            return tuple(
                TaichiCallableTemplateMapper.extract_arg(arg[name], dtype, arg_name)
                for index, (name, dtype) in enumerate(anno.members.items())
            )
        if isinstance(anno, texture_type.TextureType):
            if not isinstance(arg, taichi.lang._texture.Texture):
                raise TaichiRuntimeTypeError(f"Argument {arg_name} must be a texture, got {type(arg)}")
            if arg.num_dims != anno.num_dimensions:
                raise TaichiRuntimeTypeError(
                    f"TextureType dimension mismatch for argument {arg_name}: expected {anno.num_dimensions}, got {arg.num_dims}"
                )
            return (arg.num_dims,)
        if isinstance(anno, texture_type.RWTextureType):
            if not isinstance(arg, taichi.lang._texture.Texture):
                raise TaichiRuntimeTypeError(f"Argument {arg_name} must be a texture, got {type(arg)}")
            if arg.num_dims != anno.num_dimensions:
                raise TaichiRuntimeTypeError(
                    f"RWTextureType dimension mismatch for argument {arg_name}: expected {anno.num_dimensions}, got {arg.num_dims}"
                )
            if arg.fmt != anno.fmt:
                raise TaichiRuntimeTypeError(
                    f"RWTextureType format mismatch for argument {arg_name}: expected {anno.fmt}, got {arg.fmt}"
                )
            # (penguinliong) '0' is the assumed LOD level. We currently don't
            # support mip-mapping.
            return arg.num_dims, arg.fmt, 0
        if isinstance(anno, ndarray_type.NdarrayType):
            if isinstance(arg, taichi.lang._ndarray.Ndarray):
                anno.check_matched(arg.get_type(), arg_name)
                needs_grad = (arg.grad is not None) if anno.needs_grad is None else anno.needs_grad
                return arg.element_type, len(arg.shape), needs_grad, anno.boundary
            if isinstance(arg, AnyArray):
                ty = arg.get_type()
                anno.check_matched(arg.get_type(), arg_name)
                return ty.element_type, len(arg.shape), ty.needs_grad, anno.boundary
            # external arrays
            shape = getattr(arg, "shape", None)
            if shape is None:
                raise TaichiRuntimeTypeError(f"Invalid type for argument {arg_name}, got {arg}")
            shape = tuple(shape)
            element_shape = ()
            dtype = to_taichi_type(arg.dtype)
            if isinstance(anno.dtype, MatrixType):
                if anno.ndim is not None:
                    if len(shape) != anno.dtype.ndim + anno.ndim:
                        raise ValueError(
                            f"Invalid value for argument {arg_name} - required array has ndim={anno.ndim} element_dim={anno.dtype.ndim}, "
                            f"array with {len(shape)} dimensions is provided"
                        )
                else:
                    if len(shape) < anno.dtype.ndim:
                        raise ValueError(
                            f"Invalid value for argument {arg_name} - required element_dim={anno.dtype.ndim}, "
                            f"array with {len(shape)} dimensions is provided"
                        )
                element_shape = shape[-anno.dtype.ndim :]
                anno_element_shape = anno.dtype.get_shape()
                if None not in anno_element_shape and element_shape != anno_element_shape:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required element_shape={anno_element_shape}, "
                        f"array with element shape of {element_shape} is provided"
                    )
            elif anno.dtype is not None:
                # User specified scalar dtype
                if anno.dtype != dtype:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required array has dtype={anno.dtype.to_string()}, "
                        f"array with dtype={dtype.to_string()} is provided"
                    )

                if anno.ndim is not None and len(shape) != anno.ndim:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required array has ndim={anno.ndim}, "
                        f"array with {len(shape)} dimensions is provided"
                    )
            needs_grad = getattr(arg, "requires_grad", False) if anno.needs_grad is None else anno.needs_grad
            element_type = (
                _ti_core.get_type_factory_instance().get_tensor_type(element_shape, dtype)
                if len(element_shape) != 0
                else arg.dtype
            )
            return element_type, len(shape) - len(element_shape), needs_grad, anno.boundary
        if isinstance(anno, sparse_matrix_builder):
            return arg.dtype
        # Use '#' as a placeholder because other kinds of arguments are not involved in template instantiation
        return "#"

    def extract(self, args):
        extracted = []
        for arg, kernel_arg in zip(args, self.arguments):
            extracted.append(self.extract_arg(arg, kernel_arg.annotation, kernel_arg.name))
        return tuple(extracted)

    def lookup(self, args):
        if len(args) != self.num_args:
            raise TypeError(f"{self.num_args} argument(s) needed but {len(args)} provided.")

        key = self.extract(args)
        if key not in self.mapping:
            count = len(self.mapping)
            self.mapping[key] = count
        return self.mapping[key], key


def _get_global_vars(_func):
    # Discussions: https://github.com/taichi-dev/taichi/issues/282
    global_vars = _func.__globals__.copy()

    freevar_names = _func.__code__.co_freevars
    closure = _func.__closure__
    if closure:
        freevar_values = list(map(lambda x: x.cell_contents, closure))
        for name, value in zip(freevar_names, freevar_values):
            global_vars[name] = value

    return global_vars


class Kernel:
    counter = 0

    def __init__(self, _func, autodiff_mode, _classkernel=False):
        self.func = _func
        self.kernel_counter = Kernel.counter
        Kernel.counter += 1
        assert autodiff_mode in (
            AutodiffMode.NONE,
            AutodiffMode.VALIDATION,
            AutodiffMode.FORWARD,
            AutodiffMode.REVERSE,
        )
        self.autodiff_mode = autodiff_mode
        self.grad = None
        self.arguments = []
        self.return_type = None
        self.classkernel = _classkernel
        self.extract_arguments()
        self.template_slot_locations = []
        for i, arg in enumerate(self.arguments):
            if isinstance(arg.annotation, template):
                self.template_slot_locations.append(i)
        self.mapper = TaichiCallableTemplateMapper(self.arguments, self.template_slot_locations)
        impl.get_runtime().kernels.append(self)
        self.reset()
        self.kernel_cpp = None
        self.compiled_kernels = {}
        self.has_print = False
        self._scalar_launch_plan = _UNPLANNED  # fast path c, made at the first launch

    def ast_builder(self):
        assert self.kernel_cpp is not None
        return self.kernel_cpp.ast_builder()

    def reset(self):
        self.runtime = impl.get_runtime()
        self.compiled_kernels = {}
        # Fast path c: (program, {compiled kernel: CompiledKernelData}).  impl.reset() calls this for every kernel at
        # ti.init and ti.reset, so no kept data outlives its program (Taichi frees it when the program is finalized).
        self._kept_compiled_kernel_data = None

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect._empty, None):
            self.return_type = sig.return_annotation
            if sys.version_info >= (3, 9):
                if (
                    isinstance(self.return_type, (types.GenericAlias, typing._GenericAlias))
                    and self.return_type.__origin__ is tuple
                ):
                    self.return_type = self.return_type.__args__
            else:
                if isinstance(self.return_type, typing._GenericAlias) and self.return_type.__origin__ is tuple:
                    self.return_type = self.return_type.__args__
            if not isinstance(self.return_type, (list, tuple)):
                self.return_type = (self.return_type,)
            for return_type in self.return_type:
                if return_type is Ellipsis:
                    raise TaichiSyntaxError("Ellipsis is not supported in return type annotations")
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TaichiSyntaxError("Taichi kernels do not support variable keyword parameters (i.e., **kwargs)")
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TaichiSyntaxError("Taichi kernels do not support variable positional parameters (i.e., *args)")
            if param.default is not inspect.Parameter.empty:
                raise TaichiSyntaxError("Taichi kernels do not support default values for arguments")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise TaichiSyntaxError("Taichi kernels do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TaichiSyntaxError('Taichi kernels only support "positional or keyword" parameters')
            annotation = param.annotation
            if param.annotation is inspect.Parameter.empty:
                if i == 0 and self.classkernel:  # The |self| parameter
                    annotation = template()
                else:
                    raise TaichiSyntaxError("Taichi kernels parameters must be type annotated")
            else:
                if isinstance(
                    annotation,
                    (
                        template,
                        ndarray_type.NdarrayType,
                        texture_type.TextureType,
                        texture_type.RWTextureType,
                    ),
                ):
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif isinstance(annotation, sparse_matrix_builder):
                    pass
                elif isinstance(annotation, MatrixType):
                    pass
                elif isinstance(annotation, StructType):
                    pass
                elif isinstance(annotation, ArgPackType):
                    pass
                else:
                    raise TaichiSyntaxError(f"Invalid type annotation (argument {i}) of Taichi kernel: {annotation}")
            self.arguments.append(KernelArgument(annotation, param.name, param.default))

    def materialize(self, key=None, args=None, arg_features=None):
        if key is None:
            key = (self.func, 0, self.autodiff_mode)
        self.runtime.materialize()

        if key in self.compiled_kernels:
            return

        kernel_name = f"{self.func.__name__}_c{self.kernel_counter}_{key[1]}"
        _logging.trace(f"Compiling kernel {kernel_name} in {self.autodiff_mode}...")

        tree, ctx = _get_tree_and_ctx(
            self,
            args=args,
            excluded_parameters=self.template_slot_locations,
            arg_features=arg_features,
        )

        if self.autodiff_mode != AutodiffMode.NONE:
            KernelSimplicityASTChecker(self.func).visit(tree)

        # Do not change the name of 'taichi_ast_generator'
        # The warning system needs this identifier to remove unnecessary messages
        def taichi_ast_generator(kernel_cxx):
            if self.runtime.inside_kernel:
                raise TaichiSyntaxError(
                    "Kernels cannot call other kernels. I.e., nested kernels are not allowed. "
                    "Please check if you have direct/indirect invocation of kernels within kernels. "
                    "Note that some methods provided by the Taichi standard library may invoke kernels, "
                    "and please move their invocations to Python-scope."
                )
            self.kernel_cpp = kernel_cxx
            self.runtime.inside_kernel = True
            self.runtime.current_kernel = self
            assert self.runtime.compiling_callable is None
            self.runtime.compiling_callable = kernel_cxx
            try:
                ctx.ast_builder = kernel_cxx.ast_builder()
                transform_tree(tree, ctx)
                if not ctx.is_real_function:
                    if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                        raise TaichiSyntaxError("Kernel has a return type but does not have a return statement")
            finally:
                self.runtime.inside_kernel = False
                self.runtime.current_kernel = None
                self.runtime.compiling_callable = None

        taichi_kernel = impl.get_runtime().prog.create_kernel(taichi_ast_generator, kernel_name, self.autodiff_mode)
        assert key not in self.compiled_kernels
        self.compiled_kernels[key] = taichi_kernel

    def _make_scalar_launch_plan(self):
        """Fast path c's plan for a kernel whose non-template parameters are all primitive scalars, at most
        _MAX_ARG_NUM: (the parameter count, per scalar parameter (its position, its launch-context indices,
        _SCALAR_REAL, _SCALAR_SIGNED or _SCALAR_UNSIGNED, its annotation)); None for every other kernel.  The
        signedness is launch_kernel's is_signed(cook_dtype(annotation)): for Python's int that is the runtime's default
        integer type, signed either way."""
        setters, template_count = [], 0
        for position, argument in enumerate(self.arguments):
            needed = argument.annotation
            if isinstance(needed, template):
                template_count += 1
                continue
            if id(needed) in primitive_types.real_type_ids:
                kind = _SCALAR_REAL
            elif id(needed) in primitive_types.integer_type_ids:
                kind = _SCALAR_SIGNED if is_signed(cook_dtype(needed)) else _SCALAR_UNSIGNED
            else:
                return None
            setters.append((position, (position - template_count,), kind, needed))
        if len(setters) > _MAX_ARG_NUM:
            return None
        return len(self.arguments), tuple(setters)

    def _compiled_kernel_data(self, prog, t_kernel):
        """Fast path c: the CompiledKernelData of a compiled kernel, asked of the program once and then kept.  Taichi
        launches a compiled kernel with the data of its first compilation, also after a compile-config write such as
        autodiff's opt_level 0 -> 1 in __call__; the data is kept per program and compiled kernel."""
        kept = self._kept_compiled_kernel_data
        if kept is None or kept[0] is not prog:
            kept = self._kept_compiled_kernel_data = (prog, {})
        compiled = kept[1].get(t_kernel)
        if compiled is None:
            compiled = kept[1][t_kernel] = prog.compile_kernel(prog.config(), prog.get_device_caps(), t_kernel)
        return compiled

    def _launch_with_scalar_plan(self, t_kernel, setters, args):
        """Fast path c: launch_kernel for a kernel with a scalar plan (_make_scalar_launch_plan); the same checks,
        errors, conversions, setters, sync and return values as launch_kernel's general path."""
        launch_ctx = t_kernel.make_launch_context()
        for position, indices, kind, needed in setters:
            value = args[position]
            if kind == _SCALAR_REAL:
                if not isinstance(value, _REAL_ARGUMENT_TYPES):
                    raise TaichiRuntimeTypeError.get(indices, needed.to_string(), type(value))
                launch_ctx.set_arg_float(indices, float(value))
            elif not isinstance(value, _INTEGER_ARGUMENT_TYPES):
                raise TaichiRuntimeTypeError.get(indices, needed.to_string(), type(value))
            elif kind == _SCALAR_SIGNED:
                launch_ctx.set_arg_int(indices, int(value))
            else:
                launch_ctx.set_arg_uint(indices, int(value))
        try:
            prog = impl.get_runtime().prog
            prog.launch_kernel(self._compiled_kernel_data(prog, t_kernel), launch_ctx)
        except Exception as e:
            e = handle_exception_from_cpp(e)
            if impl.get_runtime().print_full_traceback:
                raise e
            raise e from None
        ret_dt = self.return_type
        if ret_dt is not None or self.has_print:
            runtime_ops.sync()
        if ret_dt is None:
            return None
        ret = [self.construct_kernel_ret(launch_ctx, ret_type, (i,)) for i, ret_type in enumerate(ret_dt)]
        return ret[0] if len(ret_dt) == 1 else ret

    def launch_kernel(self, t_kernel, *args):
        if _LAUNCH_FASTPATH:
            plan = self._scalar_launch_plan
            if plan is _UNPLANNED:
                plan = self._scalar_launch_plan = self._make_scalar_launch_plan()
            if plan is not None and len(args) == plan[0]:
                return self._launch_with_scalar_plan(t_kernel, plan[1], args)
        assert len(args) == len(self.arguments), f"{len(self.arguments)} arguments needed but {len(args)} provided"

        tmps = []
        callbacks = []

        actual_argument_slot = 0
        launch_ctx = t_kernel.make_launch_context()
        max_arg_num = _MAX_ARG_NUM
        exceed_max_arg_num = False

        def set_arg_ndarray(indices, v):
            v_primal = v.arr
            v_grad = v.grad.arr if v.grad else None
            if v_grad is None:
                launch_ctx.set_arg_ndarray(indices, v_primal)
            else:
                launch_ctx.set_arg_ndarray_with_grad(indices, v_primal, v_grad)

        def set_arg_texture(indices, v):
            launch_ctx.set_arg_texture(indices, v.tex)

        def set_arg_rw_texture(indices, v):
            launch_ctx.set_arg_rw_texture(indices, v.tex)

        def set_arg_ext_array(indices, v, needed):
            # Element shapes are already specialized in Taichi codegen.
            # The shape information for element dims are no longer needed.
            # Therefore we strip the element shapes from the shape vector,
            # so that it only holds "real" array shapes.
            is_soa = needed.layout == Layout.SOA
            array_shape = v.shape
            if functools.reduce(operator.mul, array_shape, 1) > np.iinfo(np.int32).max:
                warnings.warn("Ndarray index might be out of int32 boundary but int64 indexing is not supported yet.")
            if needed.dtype is None or id(needed.dtype) in primitive_types.type_ids:
                element_dim = 0
            else:
                element_dim = needed.dtype.ndim
                array_shape = v.shape[element_dim:] if is_soa else v.shape[:-element_dim]
            if isinstance(v, np.ndarray):
                if v.flags.c_contiguous:
                    launch_ctx.set_arg_external_array_with_shape(indices, int(v.ctypes.data), v.nbytes, array_shape, 0)
                elif v.flags.f_contiguous:
                    # TODO: A better way that avoids copying is saving strides info.
                    tmp = np.ascontiguousarray(v)
                    # Purpose: DO NOT GC |tmp|!
                    tmps.append(tmp)

                    def callback(original, updated):
                        np.copyto(original, np.asfortranarray(updated))

                    callbacks.append(functools.partial(callback, v, tmp))
                    launch_ctx.set_arg_external_array_with_shape(
                        indices, int(tmp.ctypes.data), tmp.nbytes, array_shape, 0
                    )
                else:
                    raise ValueError(
                        "Non contiguous numpy arrays are not supported, please call np.ascontiguousarray(arr) "
                        "before passing it into taichi kernel."
                    )
            elif has_pytorch():
                import torch  # pylint: disable=C0415

                if isinstance(v, torch.Tensor):
                    if not v.is_contiguous():
                        raise ValueError(
                            "Non contiguous tensors are not supported, please call tensor.contiguous() before "
                            "passing it into taichi kernel."
                        )
                    taichi_arch = self.runtime.prog.config().arch

                    def get_call_back(u, v):
                        def call_back():
                            u.copy_(v)

                        return call_back

                    # FIXME: only allocate when launching grad kernel
                    if v.requires_grad and v.grad is None:
                        v.grad = torch.zeros_like(v)

                    if v.requires_grad:
                        if not isinstance(v.grad, torch.Tensor):
                            raise ValueError(
                                f"Expecting torch.Tensor for gradient tensor, but getting {v.grad.__class__.__name__} instead"
                            )
                        if not v.grad.is_contiguous():
                            raise ValueError(
                                "Non contiguous gradient tensors are not supported, please call tensor.grad.contiguous() before passing it into taichi kernel."
                            )

                    tmp = v
                    if (str(v.device) != "cpu") and not (
                        str(v.device).startswith("cuda") and taichi_arch == _ti_core.Arch.cuda
                    ):
                        # Getting a torch CUDA tensor on Taichi non-cuda arch:
                        # We just replace it with a CPU tensor and by the end of kernel execution we'll use the
                        # callback to copy the values back to the original CUDA tensor.
                        host_v = v.to(device="cpu", copy=True)
                        tmp = host_v
                        callbacks.append(get_call_back(v, host_v))

                    launch_ctx.set_arg_external_array_with_shape(
                        indices,
                        int(tmp.data_ptr()),
                        tmp.element_size() * tmp.nelement(),
                        array_shape,
                        int(v.grad.data_ptr()) if v.grad is not None else 0,
                    )
                else:
                    raise TaichiRuntimeTypeError(
                        f"Argument {needed.to_string()} cannot be converted into required type {v}"
                    )
            elif has_paddle():
                import paddle  # pylint: disable=C0415

                if isinstance(v, paddle.Tensor):
                    # For now, paddle.fluid.core.Tensor._ptr() is only available on develop branch
                    def get_call_back(u, v):
                        def call_back():
                            u.copy_(v, False)

                        return call_back

                    tmp = v.value().get_tensor()
                    taichi_arch = self.runtime.prog.config().arch
                    if v.place.is_gpu_place():
                        if taichi_arch != _ti_core.Arch.cuda:
                            # Paddle cuda tensor on Taichi non-cuda arch
                            host_v = v.cpu()
                            tmp = host_v.value().get_tensor()
                            callbacks.append(get_call_back(v, host_v))
                    elif v.place.is_cpu_place():
                        if taichi_arch == _ti_core.Arch.cuda:
                            # Paddle cpu tensor on Taichi cuda arch
                            gpu_v = v.cuda()
                            tmp = gpu_v.value().get_tensor()
                            callbacks.append(get_call_back(v, gpu_v))
                    else:
                        # Paddle do support many other backends like XPU, NPU, MLU, IPU
                        raise TaichiRuntimeTypeError(f"Taichi do not support backend {v.place} that Paddle support")
                    launch_ctx.set_arg_external_array_with_shape(
                        indices, int(tmp._ptr()), v.element_size() * v.size, array_shape, 0
                    )
                else:
                    raise TaichiRuntimeTypeError(
                        f"Argument {needed.to_string()} cannot be converted into required type {v}"
                    )
            else:
                raise TaichiRuntimeTypeError(
                    f"Argument {needed.to_string()} cannot be converted into required type {v}"
                )

        def set_arg_matrix(indices, v, needed):
            if needed.dtype in primitive_types.real_types:

                def cast_func(x):
                    if not isinstance(x, (int, float, np.integer, np.floating)):
                        raise TaichiRuntimeTypeError(
                            f"Argument {needed.dtype.to_string()} cannot be converted into required type {type(x)}"
                        )
                    return float(x)

            elif needed.dtype in primitive_types.integer_types:

                def cast_func(x):
                    if not isinstance(x, (int, np.integer)):
                        raise TaichiRuntimeTypeError(
                            f"Argument {needed.dtype.to_string()} cannot be converted into required type {type(x)}"
                        )
                    return int(x)

            else:
                raise ValueError(f"Matrix dtype {needed.dtype} is not integer type or real type.")

            if needed.ndim == 2:
                v = [cast_func(v[i, j]) for i in range(needed.n) for j in range(needed.m)]
            else:
                v = [cast_func(v[i]) for i in range(needed.n)]
            v = needed(*v)
            needed.set_kernel_struct_args(v, launch_ctx, indices)

        def set_arg_sparse_matrix_builder(indices, v):
            # Pass only the base pointer of the ti.types.sparse_matrix_builder() argument
            launch_ctx.set_arg_uint(indices, v._get_ndarray_addr())

        set_later_list = []

        def recursive_set_args(needed, provided, v, indices):
            in_argpack = len(indices) > 1
            nonlocal actual_argument_slot, exceed_max_arg_num, set_later_list
            if actual_argument_slot >= max_arg_num:
                exceed_max_arg_num = True
                return 0
            actual_argument_slot += 1
            if isinstance(needed, ArgPackType):
                if not isinstance(v, ArgPack):
                    raise TaichiRuntimeTypeError.get(indices, str(needed), str(provided))
                idx_new = 0
                for j, (name, anno) in enumerate(needed.members.items()):
                    idx_new += recursive_set_args(anno, type(v[name]), v[name], indices + (idx_new,))
                launch_ctx.set_arg_argpack(indices, v._ArgPack__argpack)
                return 1
            # Note: do not use sth like "needed == f32". That would be slow.
            if id(needed) in primitive_types.real_type_ids:
                if not isinstance(v, (float, int, np.floating, np.integer)):
                    raise TaichiRuntimeTypeError.get(indices, needed.to_string(), provided)
                if in_argpack:
                    return 1
                launch_ctx.set_arg_float(indices, float(v))
                return 1
            if id(needed) in primitive_types.integer_type_ids:
                if not isinstance(v, (int, np.integer)):
                    raise TaichiRuntimeTypeError.get(indices, needed.to_string(), provided)
                if in_argpack:
                    return 1
                if is_signed(cook_dtype(needed)):
                    launch_ctx.set_arg_int(indices, int(v))
                else:
                    launch_ctx.set_arg_uint(indices, int(v))
                return 1
            if isinstance(needed, sparse_matrix_builder):
                if in_argpack:
                    set_later_list.append((set_arg_sparse_matrix_builder, (v,)))
                    return 0
                set_arg_sparse_matrix_builder(indices, v)
                return 1
            if isinstance(needed, ndarray_type.NdarrayType) and isinstance(v, taichi.lang._ndarray.Ndarray):
                if in_argpack:
                    set_later_list.append((set_arg_ndarray, (v,)))
                    return 0
                set_arg_ndarray(indices, v)
                return 1
            if isinstance(needed, texture_type.TextureType) and isinstance(v, taichi.lang._texture.Texture):
                if in_argpack:
                    set_later_list.append((set_arg_texture, (v,)))
                    return 0
                set_arg_texture(indices, v)
                return 1
            if isinstance(needed, texture_type.RWTextureType) and isinstance(v, taichi.lang._texture.Texture):
                if in_argpack:
                    set_later_list.append((set_arg_rw_texture, (v,)))
                    return 0
                set_arg_rw_texture(indices, v)
                return 1
            if isinstance(needed, ndarray_type.NdarrayType):
                if in_argpack:
                    set_later_list.append((set_arg_ext_array, (v, needed)))
                    return 0
                set_arg_ext_array(indices, v, needed)
                return 1
            if isinstance(needed, MatrixType):
                if in_argpack:
                    return 1
                set_arg_matrix(indices, v, needed)
                return 1
            if isinstance(needed, StructType):
                if in_argpack:
                    return 1
                if not isinstance(v, needed):
                    raise TaichiRuntimeTypeError(f"Argument {provided} cannot be converted into required type {needed}")
                needed.set_kernel_struct_args(v, launch_ctx, indices)
                return 1
            raise ValueError(f"Argument type mismatch. Expecting {needed}, got {type(v)}.")

        template_num = 0
        for i, val in enumerate(args):
            needed_ = self.arguments[i].annotation
            if isinstance(needed_, template):
                template_num += 1
                continue
            recursive_set_args(needed_, type(val), val, (i - template_num,))

        for i, (set_arg_func, params) in enumerate(set_later_list):
            set_arg_func((len(args) - template_num + i,), *params)

        if exceed_max_arg_num:
            raise TaichiRuntimeError(
                f"The number of elements in kernel arguments is too big! Do not exceed {max_arg_num} on {_ti_core.arch_name(impl.current_cfg().arch)} backend."
            )

        try:
            prog = impl.get_runtime().prog
            # Compile kernel (& Online Cache & Offline Cache)
            compiled_kernel_data = prog.compile_kernel(prog.config(), prog.get_device_caps(), t_kernel)
            # Launch kernel
            prog.launch_kernel(compiled_kernel_data, launch_ctx)
        except Exception as e:
            e = handle_exception_from_cpp(e)
            if impl.get_runtime().print_full_traceback:
                raise e
            raise e from None

        ret = None
        ret_dt = self.return_type
        has_ret = ret_dt is not None

        if has_ret or self.has_print:
            runtime_ops.sync()

        if has_ret:
            ret = []
            for i, ret_type in enumerate(ret_dt):
                ret.append(self.construct_kernel_ret(launch_ctx, ret_type, (i,)))
            if len(ret_dt) == 1:
                ret = ret[0]
        if callbacks:
            for c in callbacks:
                c()

        return ret

    def construct_kernel_ret(self, launch_ctx, ret_type, index=()):
        if isinstance(ret_type, CompoundType):
            return ret_type.from_kernel_struct_ret(launch_ctx, index)
        if ret_type in primitive_types.integer_types:
            if is_signed(cook_dtype(ret_type)):
                return launch_ctx.get_struct_ret_int(index)
            return launch_ctx.get_struct_ret_uint(index)
        if ret_type in primitive_types.real_types:
            return launch_ctx.get_struct_ret_float(index)
        raise TaichiRuntimeTypeError(f"Invalid return type on index={index}")

    def ensure_compiled(self, *args):
        instance_id, arg_features = self.mapper.lookup(args)
        key = (self.func, instance_id, self.autodiff_mode)
        self.materialize(key=key, args=args, arg_features=arg_features)
        return key

    # For small kernels (< 3us), the performance can be pretty sensitive to overhead in __call__
    # Thus this part needs to be fast. (i.e. < 3us on a 4 GHz x64 CPU)
    @_shell_pop_print
    def __call__(self, *args, **kwargs):
        args = _process_args(self, args, kwargs)

        # Transform the primal kernel to forward mode grad kernel
        # then recover to primal when exiting the forward mode manager
        if self.runtime.fwd_mode_manager and not self.runtime.grad_replaced:
            # TODO: if we would like to compute 2nd-order derivatives by forward-on-reverse in a nested context manager fashion,
            # i.e., a `Tape` nested in the `FwdMode`, we can transform the kernels with `mode_original == AutodiffMode.REVERSE` only,
            # to avoid duplicate computation for 1st-order derivatives
            self.runtime.fwd_mode_manager.insert(self)

        # Both the class kernels and the plain-function kernels are unified now.
        # In both cases, |self.grad| is another Kernel instance that computes the
        # gradient. For class kernels, args[0] is always the kernel owner.

        # No need to capture grad kernels because they are already bound with their primal kernels
        if (
            self.autodiff_mode in (AutodiffMode.NONE, AutodiffMode.VALIDATION)
            and self.runtime.target_tape
            and not self.runtime.grad_replaced
        ):
            self.runtime.target_tape.insert(self, args)

        if self.autodiff_mode != AutodiffMode.NONE and impl.current_cfg().opt_level == 0:
            _logging.warn("""opt_level = 1 is enforced to enable gradient computation.""")
            impl.current_cfg().opt_level = 1
        key = self.ensure_compiled(*args)
        kernel_cpp = self.compiled_kernels[key]
        return self.launch_kernel(kernel_cpp, *args)


# For a Taichi class definition like below:
#
# @ti.data_oriented
# class X:
#   @ti.kernel
#   def foo(self):
#     ...
#
# When ti.kernel runs, the stackframe's |code_context| of Python 3.8(+) is
# different from that of Python 3.7 and below. In 3.8+, it is 'class X:',
# whereas in <=3.7, it is '@ti.data_oriented'. More interestingly, if the class
# inherits, i.e. class X(object):, then in both versions, |code_context| is
# 'class X(object):'...
_KERNEL_CLASS_STACKFRAME_STMT_RES = [
    re.compile(r"@(\w+\.)?data_oriented"),
    re.compile(r"class "),
]


def _inside_class(level_of_class_stackframe):
    try:
        maybe_class_frame = sys._getframe(level_of_class_stackframe)
        statement_list = inspect.getframeinfo(maybe_class_frame)[3]
        first_statment = statement_list[0].strip()
        for pat in _KERNEL_CLASS_STACKFRAME_STMT_RES:
            if pat.match(first_statment):
                return True
    except:
        pass
    return False


def _kernel_impl(_func, level_of_class_stackframe, verbose=False):
    # Can decorators determine if a function is being defined inside a class?
    # https://stackoverflow.com/a/8793684/12003165
    is_classkernel = _inside_class(level_of_class_stackframe + 1)

    if verbose:
        print(f"kernel={_func.__name__} is_classkernel={is_classkernel}")
    primal = Kernel(_func, autodiff_mode=AutodiffMode.NONE, _classkernel=is_classkernel)
    adjoint = Kernel(_func, autodiff_mode=AutodiffMode.REVERSE, _classkernel=is_classkernel)
    # Having |primal| contains |grad| makes the tape work.
    primal.grad = adjoint

    if is_classkernel:
        # For class kernels, their primal/adjoint callables are constructed
        # when the kernel is accessed via the instance inside
        # _BoundedDifferentiableMethod.
        # This is because we need to bind the kernel or |grad| to the instance
        # owning the kernel, which is not known until the kernel is accessed.
        #
        # See also: _BoundedDifferentiableMethod, data_oriented.
        @functools.wraps(_func)
        def wrapped(*args, **kwargs):
            # If we reach here (we should never), it means the class is not decorated
            # with @ti.data_oriented, otherwise getattr would have intercepted the call.
            clsobj = type(args[0])
            if _LAUNCH_FASTPATH and any(klass in _FAST_DATA_ORIENTED for klass in clsobj.__mro__):
                # Fast path d: an instance of a class on it reached this kernel through a plain bound method, under a
                # name that held no kernel when its class was decorated (an alias set later); the 1.7.4 hook bound
                # any such name to the instance, and this calls what it bound (see data_oriented).
                wrapped._is_staticmethod = False
                return _BoundedDifferentiableMethod(args[0], wrapped)(*args[1:], **kwargs)
            assert not hasattr(clsobj, "_data_oriented")
            raise TaichiSyntaxError(f"Please decorate class {clsobj.__name__} with @ti.data_oriented")

    else:

        @functools.wraps(_func)
        def wrapped(*args, **kwargs):
            try:
                return primal(*args, **kwargs)
            except (TaichiCompilationError, TaichiRuntimeError) as e:
                if impl.get_runtime().print_full_traceback:
                    raise e
                raise type(e)("\n" + str(e)) from None

        wrapped.grad = adjoint

    wrapped._is_wrapped_kernel = True
    wrapped._is_classkernel = is_classkernel
    wrapped._primal = primal
    wrapped._adjoint = adjoint
    return wrapped


def kernel(fn):
    """Marks a function as a Taichi kernel.

    A Taichi kernel is a function written in Python, and gets JIT compiled by
    Taichi into native CPU/GPU instructions (e.g. a series of CUDA kernels).
    The top-level ``for`` loops are automatically parallelized, and distributed
    to either a CPU thread pool or massively parallel GPUs.

    Kernel's gradient kernel would be generated automatically by the AutoDiff system.

    See also https://docs.taichi-lang.org/docs/syntax#kernel.

    Args:
        fn (Callable): the Python function to be decorated

    Returns:
        Callable: The decorated function

    Example::

        >>> x = ti.field(ti.i32, shape=(4, 8))
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     # Assigns all the elements of `x` in parallel.
        >>>     for i in x:
        >>>         x[i] = i
    """
    return _kernel_impl(fn, level_of_class_stackframe=3)


class _BoundedDifferentiableMethod:
    def __init__(self, kernel_owner, wrapped_kernel_func):
        clsobj = type(kernel_owner)
        if not getattr(clsobj, "_data_oriented", False):
            raise TaichiSyntaxError(f"Please decorate class {clsobj.__name__} with @ti.data_oriented")
        self._kernel_owner = kernel_owner
        self._primal = wrapped_kernel_func._primal
        self._adjoint = wrapped_kernel_func._adjoint
        self._is_staticmethod = wrapped_kernel_func._is_staticmethod
        self.__name__ = None

    def __call__(self, *args, **kwargs):
        try:
            if self._is_staticmethod:
                return self._primal(*args, **kwargs)
            return self._primal(self._kernel_owner, *args, **kwargs)
        except (TaichiCompilationError, TaichiRuntimeError) as e:
            if impl.get_runtime().print_full_traceback:
                raise e
            raise type(e)("\n" + str(e)) from None

    def grad(self, *args, **kwargs):
        return self._adjoint(self._kernel_owner, *args, **kwargs)


def data_oriented(cls):
    """Marks a class as Taichi compatible.

    To allow for modularized code, Taichi provides this decorator so that
    Taichi kernels can be defined inside a class.

    See also https://docs.taichi-lang.org/docs/odop

    Example::

        >>> @ti.data_oriented
        >>> class TiArray:
        >>>     def __init__(self, n):
        >>>         self.x = ti.field(ti.f32, shape=n)
        >>>
        >>>     @ti.kernel
        >>>     def inc(self):
        >>>         for i in self.x:
        >>>             self.x[i] += 1.0
        >>>
        >>> a = TiArray(32)
        >>> a.inc()

    Args:
        cls (Class): the class to be decorated

    Returns:
        The decorated class.

    Fast path d of the fork (FORK.md change 7): unless TI_LAUNCH_FASTPATH=0, the class gets no __getattribute__ but a
    _DataOrientedKernel descriptor for each name whose reads the 1.7.4 hook changed (_hook_changes: its kernels, own
    or inherited from any base, and the values the hook tested) and a __setattr__ that gives a name a descriptor when
    an instance stores a kernel under it, when its instances then read every attribute as Taichi 1.7.4's hook gave it
    (_keeps_hook_semantics). A descriptor is callable, introspects and answers attribute reads as the entry it stands
    for. A class or subclass that would read differently, or a class decorated after a decorated subclass, hands its
    decorated hierarchies back to the 1.7.4 hook and takes fast path d's entries out of their classes (_revert); a class
    decorated after an undecorated subclass was made handles that subclass too. FORK.md change 7 lists the differences
    from 1.7.4; one is that a kernel set on a class after the class was made reads as a plain bound method, which runs
    the kernel for the instance when called, as the hook's bound kernel did (the safety net in _kernel_impl).
    """
    if _LAUNCH_FASTPATH and _data_oriented_fast(cls):
        return cls
    cls.__getattribute__ = _data_oriented_hook(cls)
    cls._data_oriented = True

    return cls


def _data_oriented_hook(cls):
    """Taichi 1.7.4's __getattribute__ of a data-oriented class (kernel_impl.py:1214-1237 at v1.7.4, unchanged): every
    attribute read of an instance runs it, and every kernel read builds a new _BoundedDifferentiableMethod."""

    def _getattr(self, item):
        method = cls.__dict__.get(item, None)
        is_property = method.__class__ == property
        is_staticmethod = method.__class__ == staticmethod
        if is_property:
            x = method.fget
        else:
            x = super(cls, self).__getattribute__(item)
        if hasattr(x, "_is_wrapped_kernel"):
            if inspect.ismethod(x):
                wrapped = x.__func__
            else:
                wrapped = x
            wrapped._is_staticmethod = is_staticmethod
            assert inspect.isfunction(wrapped)
            if wrapped._is_classkernel:
                ret = _BoundedDifferentiableMethod(self, wrapped)
                ret.__name__ = wrapped.__name__
                if is_property:
                    return ret()
                return ret
        if is_property:
            return x(self)
        return x

    return _getattr


# Fast path d.  Its state goes with the classes: a decorated class on fast path d -> a weak reference to the root
# (first decorated class) of its hierarchy; the classes fast path d has handled; the roots whose hierarchy went back to
# the 1.7.4 hook.
_FAST_DATA_ORIENTED = weakref.WeakKeyDictionary()
_HANDLED_CLASSES = weakref.WeakSet()
_REVERTED_ROOTS = weakref.WeakSet()
_MISSING = object()
_LOOKUP_EACH_READ = object()  # the source_entry of a descriptor that runs the 1.7.4 lookup at every read
_THIS_DESCRIPTOR = object()  # the source_entry of a descriptor made for the class __dict__ in which the lookup found it
# The interpreter's method descriptors: their reads give builtin methods, which never answer to _is_wrapped_kernel. A
# slot of __slots__ (a member descriptor) or a getset holds or computes any value, so the hook tested it.
_NATIVE_METHOD_DESCRIPTORS = (
    types.WrapperDescriptorType,
    types.MethodDescriptorType,
    types.ClassMethodDescriptorType,
)


def _delete_message():
    """The message of the AttributeError that deleting a missing instance attribute raises on this Python when the class
    has a method of that name ("name" on 3.10), as a format string with {type} and {name}."""

    class _DeleteProbe:
        def _delete_probe_name(self):
            pass

    try:
        del _DeleteProbe()._delete_probe_name
    except AttributeError as error:
        return str(error).replace("_DeleteProbe", "{type}").replace("_delete_probe_name", "{name}")
    return "{name}"


_DELETE_MESSAGE = _delete_message()


class _DataOrientedKernel:
    """Fast path d: the class attribute `name` of a data-oriented class, or of a subclass, whose reads Taichi 1.7.4's
    hook changed: a kernel (plain, staticmethod or classmethod; own or inherited) or another value that answers to
    _is_wrapped_kernel, whose read the hook bound or tested; a property or another descriptor of a class the hook did
    not special-case, whose value went through that test; or a name an instance stored such a value under
    (_install_instance_name). Reading it from an instance gives what the hook gave, reading it from the class what the
    class lookup gave, and setting and deleting do what they did (a data descriptor keeps its precedence over the
    instance __dict__). When the class __dict__ in which the lookup found the attribute holds something else now (a
    test replaced a kernel), or a class between that class and this descriptor's holds an entry of the name now (a test
    patched a mixin earlier in the MRO), a read runs the 1.7.4 lookup again, with the _is_staticmethod the hook took
    then. Where
    Class.__dict__ and inspect.getattr_static gave the entry raw in 1.7.4, they give the descriptor: it stands for raw
    there. Attribute reads it cannot answer itself go to raw (__getattr__, and the properties __doc__ and __module__,
    which every class would answer), inspect.signature and inspect.unwrap reach raw through __wrapped__, and the
    descriptor is callable when raw is (_CallableDataOrientedKernel). What only the type can answer differs (FORK.md
    change 7)."""

    # __get__ reads one slot: __getattr__ gives this class CPython's slot_tp_getattr_hook, and with it every slot read
    # loses LOAD_ATTR's cache (measured on one CPU thread: six slot reads made a kernel read 18 % slower). __get__ and
    # __set__ themselves are found on the type, so a read or a set of the class's attribute never runs __getattr__.
    __slots__ = ("name", "own", "raw", "state", "__weakref__")
    __doc__ = property(lambda self: self.__getattr__("__doc__"))
    __module__ = property(lambda self: self.__getattr__("__module__"))

    def __init__(self, name, own, source, source_entry, raw, is_staticmethod, between=()):
        self.name = name
        self.own = own  # the entry of the class it sits in that it replaced, or _MISSING
        self.raw = raw  # the class attribute as the 1.7.4 lookup found it, or _MISSING
        kind = type(raw)
        self.state = (
            name,
            source,  # the __dict__ (a live view) of the class in which the lookup found the attribute
            self if source_entry is _THIS_DESCRIPTOR else source_entry,  # what that __dict__ held for the name then
            not (hasattr(kind, "__get__") and (hasattr(kind, "__set__") or hasattr(kind, "__delete__"))),  # value first
            raw if inspect.isfunction(raw) and getattr(raw, "_is_classkernel", False) is True else None,  # a kernel
            is_staticmethod,  # the innermost decorated class's own entry is a staticmethod
            raw,
            between,  # the __dict__ views of the classes between this descriptor's class and source's, in MRO order
        )

    def __getattr__(self, attribute):
        """An attribute of raw, the entry this descriptor stands for: Python comes here for what the type and the slots
        do not answer. No such attribute where the class lookup found no entry, or on a copy whose slots are not set
        yet (copy.copy asks for __setstate__ then), which a plain read of self.raw would turn into a recursion."""
        try:
            raw = object.__getattribute__(self, "raw")
        except AttributeError:
            raise AttributeError(attribute) from None
        if raw is _MISSING:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attribute}'")
        return getattr(raw, attribute)

    @property
    def __wrapped__(self):
        """raw, the entry this descriptor stands for, so that inspect.signature and inspect.unwrap reach it as they
        reached it in Taichi 1.7.4; no such attribute where the class lookup found no entry."""
        if self.raw is _MISSING:
            raise AttributeError("__wrapped__")
        return self.raw

    def __get__(self, instance, owner=None):
        name, source, source_entry, instance_first, function, is_staticmethod, raw, between = self.state
        if instance is None:
            if owner is None:
                raise TypeError("__get__(None, None) is invalid")
            return _class_lookup(owner, name)
        # A source entry is never None: the hook changes no read of None. An entry between hides the source's.
        relookup = source.get(name) is not source_entry
        for namespace in between:
            if name in namespace and _is_entry(namespace[name]):
                relookup = True
        if relookup:
            flag = _current_flag(instance, name, is_staticmethod)
            return _hook_value(instance, _instance_lookup(instance, name), flag)
        if instance_first:
            namespace = instance.__dict__
            if name in namespace:
                return _hook_value(instance, namespace[name], is_staticmethod)
        if function is None:
            bound = raw.__get__(instance, type(instance)) if hasattr(type(raw), "__get__") else raw
            return _hook_value(instance, bound, is_staticmethod)
        function._is_staticmethod = is_staticmethod
        ret = _BoundedDifferentiableMethod(instance, function)
        ret.__name__ = function.__name__
        return ret

    def __set__(self, instance, value):
        name = self.name
        entry = _first_entry(type(instance), name)
        if _sets_and_deletes(entry):
            setter = getattr(type(entry), "__set__", None)
            if setter is None:
                raise AttributeError("__set__")  # CPython's slot for a class that defines __delete__ only
            setter(entry, instance, value)
        else:
            instance.__dict__[name] = value

    def __delete__(self, instance):
        name = self.name
        entry = _first_entry(type(instance), name)
        if _sets_and_deletes(entry):
            deleter = getattr(type(entry), "__delete__", None)
            if deleter is None:
                raise AttributeError("__delete__")  # CPython's slot for a class that defines __set__ only
            deleter(entry, instance)
            return
        try:
            del instance.__dict__[name]
        except KeyError:
            raise AttributeError(_DELETE_MESSAGE.format(type=type(instance).__name__, name=name)) from None


class _CallableDataOrientedKernel(_DataOrientedKernel):
    """Fast path d: a _DataOrientedKernel whose raw is callable (a kernel, another function, a staticmethod). Calling
    it calls raw, as a call of what Class.__dict__ or inspect.getattr_static gave did in Taichi 1.7.4; callable() is
    true for it and false for a _DataOrientedKernel, as it was for raw (a classmethod, a property)."""

    __slots__ = ()
    __doc__ = _DataOrientedKernel.__dict__["__doc__"]  # a class's own __doc__ and __module__ hide its base's
    __module__ = _DataOrientedKernel.__dict__["__module__"]

    def __call__(self, *args, **kwargs):
        return self.raw(*args, **kwargs)


def _new_descriptor(name, own, source, source_entry, raw, is_staticmethod, between=()):
    """The fast-path descriptor for name (_DataOrientedKernel's arguments): callable when raw is."""
    kind = _CallableDataOrientedKernel if callable(raw) else _DataOrientedKernel
    return kind(name, own, source, source_entry, raw, is_staticmethod, between)


def _is_entry(found):
    """Whether the class __dict__ value found is an entry as 1.7.4 had it: a fast-path descriptor that stands for an
    inherited entry is none (a subclass of a decorated class holds one for each kernel it inherits)."""
    return not isinstance(found, _DataOrientedKernel) or found.own is not _MISSING


def _current_flag(instance, name, is_staticmethod):
    """The _is_staticmethod that the 1.7.4 hook gave a value read now as name from instance: whether the own entry of
    the hook that bound it, the innermost decorated class of type(instance) (the first decorated class of its hierarchy
    on fast path d), is a staticmethod (kernel_impl.py:1217 at v1.7.4); the descriptor's own flag off fast path d."""
    layers = _fast_layers(type(instance))
    if not layers:
        return is_staticmethod
    return _own_entry(layers[-1], name).__class__ == staticmethod


def _is_dunder(name):
    """Whether name is a dunder name. Fast path d gives such a name no descriptor: one in a class __dict__ would change
    what the interpreter's own lookups of special methods find (FORK.md change 7 lists the difference)."""
    return name.startswith("__") and name.endswith("__")


def _sets_and_deletes(entry):
    """Whether the class attribute entry takes an instance's sets and deletes (its type defines __set__ or __delete__,
    which gives it CPython's tp_descr_set), as object.__setattr__ and object.__delattr__ decide."""
    kind = type(entry)
    return entry is not _MISSING and (hasattr(kind, "__set__") or hasattr(kind, "__delete__"))


def _hook_value(instance, x, is_staticmethod):
    """What the 1.7.4 hook returned for a name outside its property branch whose lookup gave x
    (kernel_impl.py:1222-1237 at v1.7.4). A descriptor that vars() or object.__setattr__ put into the instance __dict__
    answers to _is_wrapped_kernel as its entry does; it is no kernel, and reads as itself (FORK.md change 7)."""
    if hasattr(x, "_is_wrapped_kernel") and not isinstance(x, _DataOrientedKernel):
        if inspect.ismethod(x):
            wrapped = x.__func__
        else:
            wrapped = x
        wrapped._is_staticmethod = is_staticmethod
        assert inspect.isfunction(wrapped)
        if wrapped._is_classkernel:
            ret = _BoundedDifferentiableMethod(instance, wrapped)
            ret.__name__ = wrapped.__name__
            return ret
    return x


def _own_entry(klass, name):
    """klass's own __dict__ entry for name as 1.7.4 had it: a fast-path descriptor stands for the entry it replaced,
    or for no entry (_MISSING) when it stands for an inherited one."""
    found = klass.__dict__.get(name, _MISSING)
    if isinstance(found, _DataOrientedKernel):
        return found.own
    return found


def _first_entry(owner, name):
    """The first entry for name in owner.__mro__ as 1.7.4 had them, or _MISSING."""
    for klass in owner.__mro__:
        found = _own_entry(klass, name)
        if found is not _MISSING:
            return found
    return _MISSING


def _instance_lookup(instance, name):
    """object.__getattribute__(instance, name) as it ran without fast-path descriptors: a data descriptor of the
    class, then the instance __dict__, then any other class attribute (bound through its __get__)."""
    owner = type(instance)
    entry = _first_entry(owner, name)
    kind = type(entry)
    if entry is not _MISSING and hasattr(kind, "__get__") and (hasattr(kind, "__set__") or hasattr(kind, "__delete__")):
        return entry.__get__(instance, owner)
    value = instance.__dict__.get(name, _MISSING)
    if value is not _MISSING:
        return value
    if entry is _MISSING:
        raise AttributeError(f"'{owner.__name__}' object has no attribute '{name}'")
    if hasattr(kind, "__get__"):
        return entry.__get__(instance, owner)
    return entry


def _class_lookup(owner, name):
    """type.__getattribute__(owner, name) as it ran without fast-path descriptors, for a name the metaclass does not
    define."""
    entry = _first_entry(owner, name)
    if entry is _MISSING:
        raise AttributeError(f"type object '{owner.__name__}' has no attribute '{name}'")
    if hasattr(type(entry), "__get__"):
        return entry.__get__(None, owner)
    return entry


def _hook_changes(name, raw, in_layer):
    """Whether reading the class __dict__ entry raw (of a decorated class when in_layer) through an instance needs a
    descriptor to give what the 1.7.4 hook gave. Yes for a function, staticmethod or classmethod that carries
    _is_wrapped_kernel, a plain object that answers to it (a unittest.mock.Mock does), and a descriptor whose value is
    known only when it is read (a property of a class that is not decorated, a slot, another descriptor): the hook
    tested each value it read. No for a decorated class's own property (its branch of the hook is kept by
    _keeps_hook_semantics), for the interpreter's method descriptors, and for dunder names (_is_dunder)."""
    if _is_dunder(name):
        return False
    if raw.__class__ == property and in_layer:
        return False
    if isinstance(raw, (staticmethod, classmethod)):
        raw = raw.__func__
    elif isinstance(raw, _NATIVE_METHOD_DESCRIPTORS):
        return False
    elif not inspect.isfunction(raw) and hasattr(type(raw), "__get__"):
        return True
    try:
        return hasattr(raw, "_is_wrapped_kernel")
    except Exception:  # pylint: disable=broad-except
        return True  # reading it raised: the descriptor's read raises it again, where the hook raised it


def _install_kernel_descriptors(cls, layers):
    """Fast path d: a _DataOrientedKernel in cls.__dict__ for each name whose lookup in cls.__mro__ finds an entry that
    _hook_changes names, and the entry it replaced back for a name that needs none any more (cls was made undecorated
    and is decorated now). `layers` are the decorated classes of cls.__mro__; the innermost, the last, was the hook
    that bound a value, so its own entry for the name gave _is_staticmethod (kernel_impl.py:1217 at v1.7.4)."""
    found = {}
    for klass in cls.__mro__:
        for name, entry in klass.__dict__.items():
            if name in found:
                continue
            raw = entry.own if isinstance(entry, _DataOrientedKernel) else entry
            if raw is not _MISSING:
                found[name] = (klass, entry, raw)
    innermost = layers[-1]
    mro = cls.__mro__
    views = tuple(klass.__dict__ for klass in mro)
    for name, (klass, entry, raw) in found.items():
        own = _own_entry(cls, name)
        if not _hook_changes(name, raw, klass in layers):
            if isinstance(cls.__dict__.get(name), _DataOrientedKernel):
                if own is _MISSING:
                    type.__delattr__(cls, name)
                else:
                    type.__setattr__(cls, name, own)
            continue
        is_staticmethod = _own_entry(innermost, name).__class__ == staticmethod
        source_entry = _THIS_DESCRIPTOR if klass is cls else entry
        between = views[1 : mro.index(klass)]
        descriptor = _new_descriptor(name, own, klass.__dict__, source_entry, raw, is_staticmethod, between)
        type.__setattr__(cls, name, descriptor)


def _defines_getattribute(klass):
    """Whether klass defines a __getattribute__ in Python (the 1.7.4 hook or another); a builtin's slot is no such."""
    entry = klass.__dict__.get("__getattribute__", _MISSING)
    return entry is not _MISSING and not isinstance(entry, types.WrapperDescriptorType)


def _defines_setattr(klass):
    """Whether klass defines a __setattr__ in Python other than fast path d's (_data_oriented_setattr)."""
    entry = klass.__dict__.get("__setattr__", _MISSING)
    return (
        entry is not _MISSING
        and not isinstance(entry, types.WrapperDescriptorType)
        and not getattr(entry, "_data_oriented_setattr", False)
    )


def _data_oriented_setattr(root):
    """Fast path d: the __setattr__ of root, the first decorated class of a hierarchy on it. It sets as before (the next
    __setattr__ in the MRO); a descriptor as the value (what vars(Class) or inspect.getattr_static gives) stands for its
    entry, which is what 1.7.4 was given, so the entry is set. When the value went into the instance __dict__ under a
    name with no fast-path descriptor and answers to _is_wrapped_kernel (a kernel kept on the instance, a
    unittest.mock.Mock), it gives the instance's class a descriptor for that name (_install_instance_name), so that
    reading it gives what the 1.7.4 hook gave: the kernel bound to that instance, the hook's AssertionError, the hook's
    _is_staticmethod; not for a dunder name (_is_dunder). A class whose __init_subclass__ chain skipped fast path d's
    hook is handled at its instances' first set."""

    def __setattr__(self, name, value):
        cls = type(self)
        if cls not in _HANDLED_CLASSES:
            _data_oriented_subclass(cls)
        if isinstance(value, _DataOrientedKernel) and value.raw is not _MISSING:
            value = value.raw
        super(root, self).__setattr__(name, value)
        if self.__dict__.get(name, _MISSING) is not value:
            return
        try:
            binds = hasattr(value, "_is_wrapped_kernel")
        except Exception:  # pylint: disable=broad-except
            binds = True  # reading it raised: the descriptor's read raises it again, where the hook raised it
        if binds and not _is_dunder(name):
            _install_instance_name(cls, name)

    __setattr__._data_oriented_setattr = True
    return __setattr__


def _install_instance_name(cls, name):
    """Fast path d: a _DataOrientedKernel for name on cls that runs the 1.7.4 lookup at every read (an instance value
    first), unless the lookup of name in cls.__mro__ already finds one or cls's hierarchy went back to the hook. It
    stands for the class attribute that the lookup of name in cls.__mro__ finds now, or none."""
    layers = _fast_layers(cls)
    if not layers:
        return
    for klass in cls.__mro__:
        found = klass.__dict__.get(name, _MISSING)
        if found is not _MISSING:
            if isinstance(found, _DataOrientedKernel):
                return
            break
    is_staticmethod = _own_entry(layers[-1], name).__class__ == staticmethod
    descriptor = _new_descriptor(
        name, _own_entry(cls, name), cls.__dict__, _LOOKUP_EACH_READ, _first_entry(cls, name), is_staticmethod
    )
    type.__setattr__(cls, name, descriptor)


def _roots(layers):
    """The roots of the hierarchies of the decorated classes `layers` that are on fast path d."""
    return {_FAST_DATA_ORIENTED[layer]() for layer in layers if layer in _FAST_DATA_ORIENTED}


def _keeps_hook_semantics(cls, layers):
    """Whether an instance of cls reads every attribute as the 1.7.4 hooks of `layers` gave it (the decorated classes of
    cls.__mro__, cls's own included when it is being decorated) once the descriptors are in place: no other
    __getattribute__ or __setattr__ in cls.__mro__, an instance __dict__ (the descriptors keep instance attributes
    there), one decorated hierarchy, no property in cls.__mro__ whose getter answers to _is_wrapped_kernel, no layer
    property without a getter, and every property of a layer found first by the lookup (a layer's own property won
    over any other definition of its name, kernel_impl.py:1215-1221 at v1.7.4). A class's own entries are read as 1.7.4
    had them (_own_entry): __init_subclass__ puts a descriptor over a decorated subclass's own property before the
    decoration comes here."""
    if any(_defines_getattribute(klass) or _defines_setattr(klass) for klass in cls.__mro__) or cls.__dictoffset__ == 0:
        return False
    if len(_roots(layers)) > 1:
        return False
    for klass in cls.__mro__:
        for name in klass.__dict__:
            entry = _own_entry(klass, name)
            if not isinstance(entry, property):
                continue
            try:
                if hasattr(entry.fget, "_is_wrapped_kernel"):
                    return False
            except Exception:  # pylint: disable=broad-except
                return False
            if klass not in layers or entry.__class__ != property:
                continue
            if entry.fget is None:
                return False
            first = next(owner for owner in cls.__mro__ if _own_entry(owner, name) is not _MISSING)
            if first not in layers or _own_entry(first, name).__class__ != property:
                return False
    return True


def _fast_layers(cls):
    """The decorated classes of cls.__mro__ on fast path d, in MRO order."""
    return [klass for klass in cls.__mro__ if klass in _FAST_DATA_ORIENTED]


def _revert(root):
    """Hands the hierarchy of root back to Taichi 1.7.4: each of its decorated classes on fast path d gets the 1.7.4
    __getattribute__, and each of its classes gets back what fast path d replaced in its __dict__ (_take_out), so that
    the hierarchy reads, sets and introspects as in 1.7.4, also through a __getattribute__ that skips the hooks."""
    if root in _REVERTED_ROOTS:
        return
    _REVERTED_ROOTS.add(root)
    for klass in [root] + _classes_below(root):
        reference = _FAST_DATA_ORIENTED.get(klass)
        if reference is not None and reference() is root:
            del _FAST_DATA_ORIENTED[klass]
            type.__setattr__(klass, "__getattribute__", _data_oriented_hook(klass))
        _take_out(klass)


def _classes_below(cls):
    """Every class below cls, each after its bases (a base's __mro__ is shorter than its subclass's)."""
    found = set()
    pending = [cls]
    while pending:
        for subclass in type.__subclasses__(pending.pop()):
            if subclass not in found:
                found.add(subclass)
                pending.append(subclass)
    return sorted(found, key=lambda klass: len(klass.__mro__))


def _take_out(klass):
    """Puts back into klass.__dict__ what fast path d replaced there: each descriptor's own entry (or no entry), the entry
    it stands for where a descriptor was copied under another name, and the entry that fast path d's __setattr__ or
    __init_subclass__ hook replaced (or none)."""
    for name, entry in list(klass.__dict__.items()):
        if isinstance(entry, _DataOrientedKernel):
            kept = entry.own if entry.name == name else entry.raw
            if kept is _MISSING:
                type.__delattr__(klass, name)
            else:
                type.__setattr__(klass, name, kept)
    if getattr(klass.__dict__.get("__setattr__"), "_data_oriented_setattr", False):
        type.__delattr__(klass, "__setattr__")
    hook = getattr(klass.__dict__.get("__init_subclass__"), "__func__", None)
    if getattr(hook, "_data_oriented_subclass_hook", False):
        if hook.previous is _MISSING:
            type.__delattr__(klass, "__init_subclass__")
        else:
            type.__setattr__(klass, "__init_subclass__", hook.previous)


def _revert_all(layers):
    """Hands the hierarchies of the decorated classes `layers` back to Taichi 1.7.4 (_revert)."""
    for root in _roots(layers):
        _revert(root)


def _data_oriented_fast(cls):
    """Fast path d when cls is decorated: descriptors and no hook when cls keeps the 1.7.4 semantics with them and has
    no decorated subclass on fast path d yet (True); otherwise the decorated hierarchies of cls.__mro__ and of its
    subclasses go back to the 1.7.4 hook and cls gets it too (False): cls's hook would run under its subclass's, and
    it, the innermost, bound each value, with cls's own flag (kernel_impl.py:1217 at v1.7.4). Subclasses made before cls
    was decorated are handled as if they were made now."""
    inherited = [klass for klass in _fast_layers(cls) if klass is not cls]
    layers = [cls] + inherited
    below = _classes_below(cls)
    _HANDLED_CLASSES.add(cls)
    if not _keeps_hook_semantics(cls, layers) or any(klass in _FAST_DATA_ORIENTED for klass in below):
        _revert_all(layers + below)
        return False
    root = _FAST_DATA_ORIENTED[inherited[0]]() if inherited else cls
    _FAST_DATA_ORIENTED[cls] = weakref.ref(root)
    _install_kernel_descriptors(cls, layers)
    cls._data_oriented = True
    if root is cls:
        _install_subclass_hook(cls)
        type.__setattr__(cls, "__setattr__", _data_oriented_setattr(cls))
    for subclass in below:
        _data_oriented_subclass(subclass)
    return True


def _data_oriented_subclass(subclass):
    """Fast path d when a subclass of a decorated class on it is made: descriptors for the subclass when it keeps the
    1.7.4 semantics with them; otherwise its decorated hierarchies go back to the 1.7.4 hook. A subclass that brings its
    own __init_subclass__ gets this hook around it, so that its subclasses come here too."""
    _HANDLED_CLASSES.add(subclass)
    layers = _fast_layers(subclass)
    if not layers:
        return
    if "__init_subclass__" in subclass.__dict__:
        _install_subclass_hook(subclass)
    if not _keeps_hook_semantics(subclass, layers):
        _revert_all(layers)
        return
    _install_kernel_descriptors(subclass, layers)


def _install_subclass_hook(cls):
    """Fast path d: cls.__init_subclass__ runs _data_oriented_subclass for each new subclass, after cls's own
    __init_subclass__ or else the next one in the MRO, which it keeps."""
    previous = cls.__dict__.get("__init_subclass__", _MISSING)
    if getattr(getattr(previous, "__func__", None), "_data_oriented_subclass_hook", False):
        return

    def __init_subclass__(subclass, **kwargs):
        if previous is _MISSING:
            super(cls, subclass).__init_subclass__(**kwargs)
        else:
            previous.__get__(None, subclass)(**kwargs)
        _data_oriented_subclass(subclass)

    __init_subclass__._data_oriented_subclass_hook = True
    __init_subclass__.previous = previous  # what _take_out puts back
    type.__setattr__(cls, "__init_subclass__", classmethod(__init_subclass__))


__all__ = ["data_oriented", "func", "kernel", "pyfunc", "real_func"]
