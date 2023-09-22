import typing
import shlex
import inspect
import types

import pydoc

from util.type_utils import type_name, is_generic_instance

def defer(subnode):
    return subnode()

T = typing.TypeVar("T")
class Factory(typing.Generic[T]):
    def __init__(self, type_or_typename : typing.Union[type, str, None] = None, *args, **kwargs):
        if isinstance(type_or_typename, str):
            self.func_type = pydoc.locate(type_or_typename)
            if self.func_type is None:
                raise Exception(f"No such class or function found {type_or_typename} during Factory.init (are you missing the module name or import?)")
        else:
            self.func_type = type_or_typename

        if self.func_type is not None and not callable(self.func_type) and not isinstance(self.func_type, type):
            raise Exception(f"Factory requires a callable function or class but got: {self.func_type}")

        self.args = args
        self.kwargs = dict(kwargs)
        self.replace_with_instance = False

    def __call__(self, /, *args, **kwargs):
        if self.func_type is None:
            return None

        def recursively_replace_factory_as_needed(c):
            if isinstance(c, list):
                for i, v in enumerate(c):
                    c[i] = recursively_replace_factory_as_needed(v)
            elif isinstance(c, dict):
                for k, v in c.items():
                    c[k] = recursively_replace_factory_as_needed(v)
            elif isinstance(c, Factory) and c.replace_with_instance:
                c = c()
            return c

        args = [*self.args, *args]
        # FIXME - is this the kind of merge we want?
        kwargs = {**self.kwargs, **kwargs}
        #kwargs = dict(self.kwargs)
        #kwargs.update(kwargs)

        for i, v in enumerate(args):
            args[i] = recursively_replace_factory_as_needed(v)
        for k, v in kwargs.items():
            kwargs[k] = recursively_replace_factory_as_needed(v)
       
        return self.func_type(*args, **kwargs)
    
    def toDict(self):
        def _process(node):
            match node:
                case Factory():
                    return {"class_path":node.func_type, "pos_init_args":_process(node.args), "init_args":_process(node.kwargs)}
                case list():
                    return [map(_process, node)]
                case dict():
                    return {key : _process(value) for key, value in node.items()}
                case set():
                    return set(map(_process, node))
        return _process(self)
    
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        rv = f"{type(self).__name__}({self.func_type}"
        argstr = ','.join(['{!r}'.format(v) for v in self.args])
        if len(argstr) > 0:
            rv += ', ' + argstr
        kwstr = ', '.join(['{}={!r}'.format(k, v) for k, v in self.kwargs.items()])
        if len(kwstr) > 0:
            rv += ', ' + kwstr
        rv += ')'
        return rv    
    def __contains__(self, key):
        return key in self.kwargs
    def __len__(self) -> int: return len(self.args) + len(self.kwargs)
    def __getitem__(self, __key: str) -> typing.Any: return self.kwargs[__key]
    def __setitem__(self, __key: str, __value: typing.Any) -> None: self.kwargs[__key] = __value
    def __delitem__(self, __key: str) -> None: del self.kwargs[__key]
    # def __iter__(self) -> Iterator[str]: ...

import collections
collections.UserDict

def RFactory(type_or_typename : typing.Union[type, str, None] = None, *args, **kwargs):
    rv = Factory(type_or_typename, *args, **kwargs)
    rv.replace_with_instance = True
    return rv

def merge(dst : typing.Union[Factory, dict], src : typing.Union[Factory, dict]):
    # merge instances of Factory, dict, and list
    if type(src) == type(dst):
        if isinstance(src, Factory):
            # FIXME - add some sort of required superclass too
            if src.func_type != type(None):
                dst.func_type = src.func_type
            #dst.args = src.args
            dst.kwargs = merge(dst.kwargs, src.kwargs)
            return dst
        elif isinstance(src, dict):
            for k, srcv in src.items():
                dst[k] = merge(dst[k], srcv) if k in dst else srcv
            return dst
        elif isinstance(src, list):
            dst.clear()
            for i, srcv in enumerate(src):
                dst[k] = merge(dst[k], srcv)
            return dst
    # otherwise, just copy it
    return src

def typecheck(path : str, self : typing.Any, required_type : type = typing.Any):
    try:
        #print(f"typecheck {path}.{parent_key} {required_type}")

        errors = ''

        #if type == typing.Any:
        #    return errors

        if not is_generic_instance(self, required_type):
            if not isinstance(self, Factory):
                #    print(f"Factory found {v.func_type}")
                # get_config_func_options(parent_key, value) + \
                errors += f"Type Mismatch: expected {type_name(required_type)} but got {type_name(type(self))}\n in config setting `{path}` : {type_name(required_type)} = {self}\n"
                return errors

        if isinstance(self, Factory):
            # fill in func_type when Empty but we have a type annotation
            # FIXME - add and check some sort of required superclass too
            if self.func_type == type(None):
                self.func_type = required_type

            if isinstance(self.func_type, types.FunctionType):
                sig = inspect.signature(self.func_type)
            else:
                sig = inspect.signature(self.func_type.__init__)

            for k, p in sig.parameters.items():
                #if kind in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD):
                if k in self.kwargs:
                    continue

                if p.default is not p.empty:
                    continue

                if p.annotation is p.empty:
                    continue
                
                # FIXME - had to disable checking for missing config settings, because we purposely pass some manually
                # if (p.annotation != typing.Any and typing.get_origin(p.annotation) != typing.Optional) and not (typing.get_origin(p.annotation) == typing.Union and len(typing.get_args(p.annotation)) == 2 and typing.get_args(p.annotation)[1]==type(None)):
                #     return f"Missing config setting `{path}.{k}` : {type_name(p.annotation)}\n"

            # traverse all subelements recursively
            for k, v in self.kwargs.items():
                if k not in sig.parameters.keys():
                    return f'Disallowed config entry `{path}.{k}` - No such parameter {k} in {self.func_type}\n'
                p = sig.parameters[k]
                rt = p.annotation
                if rt == inspect.Parameter.empty:
                    rt = typing.Any
                errors += typecheck(path + '.' + k, v, rt)


        elif isinstance(self, dict):
            # traverse all subelements recursively
            for k, v in self.items():
                errors += typecheck(path + '.' + k, v)
        elif isinstance(self, list):
            # traverse all subelements recursively
            for i, v2 in enumerate(self):
                errors += typecheck(f'{path}[{i}]', v2)

    except Exception as ex:
        errors += f'internal config exception at path "{path}": {self} {required_type} {ex}'


    return errors            


from _ast import *
import sys
import ast

class ConfigParseError(Exception):
    def __init__(self, node, unparsed_input, msg):
        lineno = getattr(node, 'lineno', None)
        end_lineno = getattr(node, 'end_lineno', None)
        if lineno is not None:
            lines = unparsed_input.split('\n')
            if end_lineno is not None and end_lineno != node.lineno:
                end_col_offset = len(lines[node.lineno]) + 1
            else:
                end_col_offset = node.end_col_offset + 1
            msg += f' at line {lineno}, col {node.col_offset+1}\n'
            msg += lines[lineno-1] + '\n'
            msg += ' '*node.col_offset
            msg += '^'*(end_col_offset - node.col_offset) + '\n'
        super().__init__(msg)


class ConfigParser():
    def eval_first_expr(self, unparsed_input : str):
        self.unparsed_input = unparsed_input
        self.imports_map = {}
        node = ast.parse(unparsed_input)
        if isinstance(node, Module):
            # get the first expression in the module
            node = node.body
            for subnode in node:
                if isinstance(subnode, Import):
                    for alias in subnode.names:
                        if alias.asname is not None:
                            self.imports_map[alias.asname] = alias.name
                elif isinstance(subnode, ImportFrom):
                    for alias in subnode.names:
                        self.imports_map[alias.asname if alias.asname is not None else alias.name] = subnode.module + '.' + alias.name
                elif isinstance(subnode, Expr):
                    node = subnode.value
                    break
        if isinstance(node, Expression):
            node = node.body
        return self.process(node)

    def process(self, node):
        try:
            match node:
                case Constant():
                    return node.value
                case List():
                    return list(map(self.process, node.elts))
                case Set():
                    return set(map(self.process, node.elts))
                case Tuple():
                    return tuple(map(self.process, node.elts))
                case Dict():
                    if len(node.keys) != len(node.values):
                        raise ConfigParseError(node, self.unparsed_input, "dictionary data had different number of keys and values")
                    return dict(zip(map(self.process, node.keys), map(self.process, node.values)))
                case BinOp():
                    if isinstance(node.op, (Add, Sub, Mult, Mod, Div, FloorDiv, Pow, RShift, LShift, BitAnd, BitOr, BitXor, And, Or)):
                        left = self.process(node.left)
                        right = self.process(node.right)
                        if isinstance(node.op, Add):
                            return left + right
                        if isinstance(node.op, Sub):
                            return left - right
                        elif isinstance(node.op, Mult):
                            return left * right
                        elif isinstance(node.op, Mod):
                            return left % right
                        elif isinstance(node.op, Div):
                            return left / right
                        elif isinstance(node.op, FloorDiv):
                            return left // right
                        elif isinstance(node.op, Pow):
                            return left ** right
                        elif isinstance(node.op, RShift):
                            return left >> right
                        elif isinstance(node.op, LShift):
                            return left << right
                        elif isinstance(node.op, BitAnd):
                            return left & right
                        elif isinstance(node.op, BitOr):
                            return left | right
                        elif isinstance(node.op, BitXor):
                            return left ^ right
                        elif isinstance(node.op, And):
                            return left and right
                        elif isinstance(node.op, Or):
                            return left or right
                        raise Exception(f"binary operations allowed only on numeric constants")
                case UnaryOp():
                    if isinstance(node.op, (Invert, Not, UAdd, USub)):
                        operand = self.process(node.operand)
                        if isinstance(node.op, Invert):
                            return ~operand
                        elif isinstance(node.op, Not):
                            return not operand
                        elif isinstance(node.op, UAdd):
                            return +operand
                        elif isinstance(node.op, USub):
                            return -operand
                case Attribute():
                    id = self.process(node.value) + '.' + str(node.attr)
                    return self.imports_map[id] if id in self.imports_map else id
                case Name():
                    id = node.id
                    rv = self.imports_map[id] if id in self.imports_map else id
                    return rv
                case Lambda():
                    if len(node.args.args) > 0 or len(node.args.kwonlyargs) > 0 or not isinstance(node.body, Call) or not isinstance(node.body.func, (Name, Attribute)):
                        raise Exception('configuration lambda must have zero arguments, and is used to return a Factory')
                    node = node.body
                    rv = self.create_factory(node, *map(self.process, node.args), **{kw.arg:self.process(kw.value) for kw in node.keywords})
                    return rv
                case Call():
                    if isinstance(node.func, (Name, Attribute)):
                        id = self.process(node.func)
                        match id:
                            case 'str' | 'float' | 'int' | 'bool':
                                return getattr(sys.modules['builtins'], id)(self.process(node.args[0]))
                            case 'list':
                                return list(map(self.process, node.args))
                            case 'set' :
                                return set(map(self.process, node.args))
                            case 'dict':
                                return {k.arg : self.process(k.value) for k in node.keywords}
                            case 'config.Factory':
                                return self.create_factory(node.args[0], *map(self.process, node.args[1:]), **{kw.arg:self.process(kw.value) for kw in node.keywords})
                            case _:
                                # allows UDTs to be created via deferred instantiation
                                rv = self.create_factory(node, *map(self.process, node.args), **{kw.arg:self.process(kw.value) for kw in node.keywords})
                                rv.replace_with_instance = True
                                return rv
        except ConfigParseError:
            raise
        except Exception as e:
            raise
        #    raise ConfigParseError(node, self.unparsed_input, "") from e
        raise ConfigParseError(node, self.unparsed_input, msg = 'unsupported language element')

    def create_factory(self, node, *args, **kwargs):
        func_node = node.func
        func_name = self.process(func_node)
        func_type = pydoc.locate(func_name)
        if func_type is None:
            raise ConfigParseError(node, self.unparsed_input, f"No such class or function found {func_name} while constructing Factory (are you missing the module name or import?)")
        if not callable(func_type) and not isinstance(func_type, type):
            raise ConfigParseError(node, self.unparsed_input, f"Factory requires a callable function or class but got: {func_name}")
        try:
            rv = Factory(func_type, *args, **kwargs)
        except Exception as e:
            raise ConfigParseError(node, self.unparsed_input, str(e)) from None
        return rv

def eval_first_expr(input : str):
    return ConfigParser().eval_first_expr(input)
