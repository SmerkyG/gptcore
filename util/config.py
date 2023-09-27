import typing
import types
import shlex
import inspect
import types
import collections.abc

import pydoc

import builtins
class MissingType(): pass
Missing = MissingType()
# same as pydoc.locate, except this function can tell you if an attribute exists even if it's set to None
def locate(path, missing_value=None, forceload:bool=False):
    """Locate an object by name or dotted path, importing as necessary."""
    parts = [part for part in path.split('.') if part]
    module, n = None, 0
    while n < len(parts):
        nextmodule = pydoc.safeimport('.'.join(parts[:n+1]), forceload)
        if nextmodule: module, n = nextmodule, n + 1
        else: break
    if module:
        object = module
    else:
        object = builtins
    for part in parts[n:]:
        try:
            object = getattr(object, part)
        except AttributeError:
            return missing_value
    return object


from util.type_utils import type_name, is_generic_instance

T = typing.TypeVar("T")
class IPartial(typing.Generic[T]):
    # FIXME - put replace_with_instance here instead of in each subclass?
    pass

def recursively_replace_factory_as_needed(c):
    if isinstance(c, list):
        for i, v in enumerate(c):
            c[i] = recursively_replace_factory_as_needed(v)
    elif isinstance(c, dict):
        for k, v in c.items():
            c[k] = recursively_replace_factory_as_needed(v)
    elif isinstance(c, Factory) and c.replace_with_instance:
        c = c()
    elif isinstance(c, MemberAccessor) and c.replace_with_instance:
        c = c()
    elif isinstance(c, IdentifierAccessor) and c.replace_with_instance:
        c = c()
    return c

def recursively_replace_identifier_accessors_as_needed(c):
    if isinstance(c, list):
        for i, v in enumerate(c):
            c[i] = recursively_replace_identifier_accessors_as_needed(v)
    elif isinstance(c, dict):
        for k, v in c.items():
            c[k] = recursively_replace_identifier_accessors_as_needed(v)
    elif isinstance(c, Factory):
        for i, v in enumerate(c.args):
            c.args[i] = recursively_replace_identifier_accessors_as_needed(v)
        for k, v in c.kwargs.items():
            c.kwargs[k] = recursively_replace_identifier_accessors_as_needed(v)
    elif isinstance(c, MemberAccessor):
        c.inner_ipartial = recursively_replace_identifier_accessors_as_needed(c.inner_ipartial)
        for i, v in enumerate(c.args):
            c.args[i] = recursively_replace_identifier_accessors_as_needed(v)
        for k, v in c.kwargs.items():
            c.kwargs[k] = recursively_replace_identifier_accessors_as_needed(v)
    elif isinstance(c, IdentifierAccessor):
        #c_old = c
        c = c()
        #print(f"Replaced IdentifierAccessor {c_old} with {c}")
    return c

class Factory(typing.Generic[T], IPartial[T]):
    def __init__(self, type_or_typename : typing.Union[type, str, None] = None, placeholders={}, *args, **kwargs):
        if isinstance(type_or_typename, str):
            self.func_type = pydoc.locate(type_or_typename)
            if self.func_type is None:
                raise Exception(f"No such class or function found {type_or_typename} during Factory.init (are you missing the module name or import?)")
        else:
            self.func_type = type_or_typename

        if self.func_type is not None and not callable(self.func_type) and not isinstance(self.func_type, type):
            raise Exception(f"Factory requires a callable function or class but got: {self.func_type}")

        self.args = list(args)
        self.kwargs = dict(kwargs)
        self.placeholders = dict(placeholders)
        self.replace_with_instance = False

    def __call__(self, /, *args, **kwargs) -> T:
        if self.func_type is None:
            return None

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
        if self is None:
            return None
        def _process(node):
            match node:
                case Factory():
                    rv = {"class_path":type_name(node.func_type)}
                    if len(node.args) > 0: rv["pos_init_args"] = _process(node.args)
                    if len(node.kwargs) > 0: rv["init_args"] = _process(node.kwargs)
                    return rv
                case list():
                    return list(map(_process, node))
                case dict():
                    return {key : _process(value) for key, value in node.items()}
                case set():
                    return set(map(_process, node))
                case str():
                    return shlex.quote(node)
                case _:
                    return str(node)
        return _process(self)
    
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        rv = f"{type(self).__name__}({type_name(self.func_type)}"
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


class IdentifierAccessor(typing.Generic[T], IPartial[T]):
    def __init__(self, fullid : str, replace_with_instance:bool = True):
        #print(f"IdentifierAccessor {fullid}")
        self.fullid = fullid
        self.replace_with_instance = replace_with_instance

    def __call__(self) -> typing.Any: # can't know the return type, unfortunately, since it's a string based identifier accessor
        rv = pydoc.locate(self.fullid)
        #print(f"Calling IdentifierAccessor {self.fullid} == {rv}")
        return rv

    def __repr__(self):
        return f"{type(self).__name__}({self.fullid})"

class MemberAccessor(typing.Generic[T], IPartial[T]):
    def __init__(self, member_name : str, inner_ipartial : IPartial[T], is_call:bool = False, replace_with_instance:bool = True, placeholders={}, *args, **kwargs):
        self.inner_ipartial = inner_ipartial
        self.member_name = member_name
        self.is_call = is_call
        self.args = list(args)
        self.kwargs = dict(kwargs)
        self.placeholders = dict(placeholders)
        self.replace_with_instance = replace_with_instance

    def __call__(self, /, *args, **kwargs) -> typing.Any: # can't know the return type, unfortunately, since it's a string based property accessor or member fn invocation
        #print(f"MEMBERACCESSOR {self.inner_ipartial}", self.is_call, args, self.args, kwargs, self.kwargs)

        args = [*self.args, *args]

        # FIXME - is this the kind of merge we want?
        kwargs = {**self.kwargs, **kwargs}
        kwargs = dict(self.kwargs)
        kwargs.update(kwargs)

        #for i, v in enumerate(args):
        #    args[i] = recursively_replace_factory_as_needed(v)
        #for k, v in kwargs.items():
        #    kwargs[k] = recursively_replace_factory_as_needed(v)

        obj = recursively_replace_factory_as_needed(self.inner_ipartial)

        # must be a property OR could be a weird override like in IterDataPipe - they manually override __getattr__ to return partials corresponding to shuffle, etc.
        attr = getattr(obj, self.member_name, None)
        #print(f"MEMBER attr '{self.member_name}' {attr}")
        if self.is_call:
            #if not callable(attr):
            #    raise 
            attr = attr(*self.args, **self.kwargs)
            #print(f"CALLABLE CALLED {attr}")
        else:
            #print(f"UNCALLABLE {attr}")
            pass
        return attr

    def __repr__(self):
        rv = f"{type(self).__name__}({self.inner_ipartial}, {self.member_name}"
        if self.is_call:
            argstr = ','.join(['{!r}'.format(v) for v in self.args])
            if len(argstr) > 0:
                rv += ', ' + argstr
            kwstr = ', '.join(['{}={!r}'.format(k, v) for k, v in self.kwargs.items()])
            if len(kwstr) > 0:
                rv += ', ' + kwstr
        rv += ')'
        return rv    

import collections
collections.UserDict

def RFactory(type_or_typename : type | str | None = None, *args, **kwargs):
    rv = Factory(type_or_typename, *args, **kwargs)
    rv.replace_with_instance = True
    return rv

def merge(dst : Factory | dict, src : Factory | dict):
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
                dst[i] = merge(dst[i], srcv)
            return dst
    # otherwise, just copy it
    return src

def typecheck(path : str, self : typing.Any, required_type : type = typing.Any, is_replace_with_instance_factory:bool = True):
    errors = ''
    try:
        # print(f"typecheck {path} {required_type}")
        if isinstance(required_type, str):
            # if the required type is a string (which could happen due to weird python pre-declarations that aren't available, and is done in lightning.LightningModule.fit's model parameter type)
            # then just allow anything through, since we can't realistically type check this
            required_type = typing.Any

        #if type == typing.Any:
        #    return errors

        if isinstance(self, IdentifierAccessor):
            # allow IdentifierAccessors through, since they return Any
            return errors


        if not is_generic_instance(self, required_type):           
            if not isinstance(self, Factory):
                #    print(f"Factory found {v.func_type}")
                # get_config_func_options(parent_key, value) + \
                errors += f"Config Type Mismatch: expected {type_name(required_type)} but got {type_name(type(self))}\n in config setting `{path}` : {type_name(required_type)} = {self}\n"
                return errors

        if isinstance(self, Factory):
            if typing.get_origin(required_type) == collections.abc.Callable and self.replace_with_instance: # wanted a factory but got what is probably a method invocation
                errors += f"Config Type Mismatch: expected Factory/Callable but got immediate invocation of {type_name(self.func_type)} (did you forget to prefix this with 'lambda:'?)\n in config setting `{path}` : {type_name(required_type)} = {self}\n"
                return errors

            # fill in func_type when Empty but we have a type annotation
            # FIXME - add and check some sort of required superclass too
            if self.func_type == type(None):
                self.func_type = required_type

            if isinstance(self.func_type, types.FunctionType):
                sig = inspect.signature(self.func_type)
                is_fn = True
            else:
                sig = inspect.signature(self.func_type.__init__)
                is_fn = False

            for k, p in sig.parameters.items():
                #if kind in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD):
                if k in self.kwargs or k in self.placeholders:
                    continue

                if p.default is not p.empty:
                    continue

                if p.annotation is p.empty:
                    continue

                # FIXME - had to disable checking for missing config settings, because we purposely pass some manually
                if (p.annotation != typing.Any and typing.get_origin(p.annotation) != typing.Optional):
                    if not (typing.get_origin(p.annotation) in [typing.Union, types.UnionType] and len(typing.get_args(p.annotation)) == 2 and typing.get_args(p.annotation)[1]==type(None)):
                        return f"Missing config setting `{path}.{k}` : {type_name(p.annotation)}\n"

            # traverse all subelements recursively
            for k, v in self.kwargs.items():
                if k not in sig.parameters.keys():
                    return f'Disallowed config entry `{path}.{k}` - No such parameter {k} in {self.func_type}\n'
                p = sig.parameters[k]
                rt = p.annotation
                if rt == inspect.Parameter.empty:
                    rt = typing.Any
                errors += typecheck(path + '.' + k, v, rt, self.replace_with_instance)


        elif isinstance(self, dict):
            # traverse all subelements recursively
            for k, v in self.items():
                errors += typecheck(path + '.' + k, v)
        elif isinstance(self, list):
            # traverse all subelements recursively
            for i, v2 in enumerate(self):
                errors += typecheck(f'{path}[{i}]', v2)

    except Exception as ex:
        raise Exception(f'internal config type checking exception at path "{path}": {required_type} {ex}')


    return errors            


from _ast import *
import sys
import ast

class ConfigInstantiationError(Exception):
    pass

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

class LocalIdentifier():
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"LocalIdentifier('{self.name}')"

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
                        self.imports_map[alias.asname if alias.asname is not None else alias.name] = '.'.join([str(subnode.module), alias.name])
                elif isinstance(subnode, Expr):
                    node = subnode.value
                    break
        if isinstance(node, Expression):
            node = node.body
        self.locals = {}
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
                        # FIXME - check that left and right are both numeric or boolean (or string for +) instead of letting it cause an exception
                        match node.op:
                            case Add():
                                return left + right
                            case Sub():
                                return left - right
                            case Mult():
                                return left * right
                            case Mod():
                                return left % right
                            case Div():
                                return left / right
                            case FloorDiv():
                                return left // right
                            case Pow():
                                return left ** right
                            case RShift():
                                return left >> right
                            case LShift():
                                return left << right
                            case BitAnd():
                                return left & right
                            case BitOr():
                                return left | right
                            case BitXor():
                                return left ^ right
                            case And():
                                return left and right
                            case Or():
                                return left or right
                    raise ConfigParseError(node, self.unparsed_input, msg = 'unsupported binary operation')
                case UnaryOp():
                    if isinstance(node.op, (Invert, Not, UAdd, USub)):
                        operand = self.process(node.operand)
                        match(node.op):
                            case Invert():
                                return ~operand
                            case Not():
                                return not operand
                            case UAdd():
                                return +operand
                            case USub():
                                return -operand
                    raise ConfigParseError(node, self.unparsed_input, msg = 'unsupported unary operation')
                case Attribute():
                    value = self.process(node.value)
                    if not isinstance(value, IdentifierAccessor):
                        #print(value)
                        #print(ast.dump(node))
                        return MemberAccessor(member_name=node.attr, inner_ipartial=value, is_call=False, replace_with_instance=True)
                        #raise ConfigParseError(node, self.unparsed_input, 'configuration files do not support member access (neither properties nor functions)')
                    id = value.fullid + '.' + str(node.attr)
                    fullid = self.imports_map[id] if id in self.imports_map else id

                    located = locate(fullid, Missing)
                    if located is Missing:
                        raise ConfigParseError(node, self.unparsed_input, f"could not locate identifier '{id}'")
                    return IdentifierAccessor(fullid)
                case Name():
                    id = node.id

                    # check if it's a local
                    if id in self.locals:
                        return LocalIdentifier(id)

                    fullid = self.imports_map[id] if id in self.imports_map else id
                    return IdentifierAccessor(fullid)
                case Lambda():
                    locals_tmp = self.locals
                    # FIXME - this next commented line disallows lambda args
                    # if len(node.args.posonlyargs) > 0 or len(node.args.args) > 0 or len(node.args.kwonlyargs) > 0:                    
                    if len(node.args.args) > 0:
                        self.locals = self.locals.copy()
                        for arg in node.args.args: self.locals[arg.arg] = True

                    if not isinstance(node.body, Call) or not isinstance(node.body.func, (Name, Attribute)):
                        raise ConfigParseError(node, self.unparsed_input, 'configuration lambda must be used to return a Factory or MemberAccessor')
                        #raise ConfigParseError(node, self.unparsed_input, 'configuration lambda must have zero arguments, and is used to return a Factory')
                    # FIXME - instead of calling create_factory directly here we could somehow indicate that the result should be replace_with_instance=False and just always process node.body.func
                    node = node.body
                    if isinstance(node.func, Attribute) and isinstance(node.func.value, Call):
                        # create MemberAccessor
                        args, kwargs, placeholders = self.process_args_and_keywords(node_args=node.args, node_keywords=node.keywords)
                        #print("CAPTURING MemberAccess ", node.func.attr, placeholders, args, kwargs)
                        return MemberAccessor(member_name=node.func.attr, inner_ipartial=self.process(node.func.value), is_call=True, replace_with_instance=False, placeholders=placeholders, *args, **kwargs)
                    else:
                        # create Factory
                        rv = self.create_factory(node, node.args, node.keywords, replace_with_instance=False)

                    self.locals = locals_tmp
                    return rv
                case Call():
                    if isinstance(node.func, Attribute) and isinstance(node.func.value, Call):
                        #print()
                        #print(ast.dump(node))
                        #print()
                        # create MemberAccessor
                        args, kwargs, placeholders = self.process_args_and_keywords(node_args=node.args, node_keywords=node.keywords)
                        #print("CAPTURING CallMemberAccess ", node.func.attr, placeholders, args, kwargs)
                        return MemberAccessor(member_name=node.func.attr, inner_ipartial=self.process(node.func.value), is_call=True, replace_with_instance=True, placeholders=placeholders, *args, **kwargs)

                    if isinstance(node.func, (Name, Attribute)):
                        ident = self.process(node.func)
                        match ident.fullid:
                            case 'str' | 'float' | 'int' | 'bool':
                                return getattr(sys.modules['builtins'], ident.name)(self.process(node.args[0]))
                            case 'list':
                                return list(map(self.process, node.args))
                            case 'set' :
                                return set(map(self.process, node.args))
                            case 'dict':
                                return {k.arg : self.process(k.value) for k in node.keywords}
                            case 'config.Factory':
                                return self.create_factory(node.args[0], node.args[1:], node.keywords)
                            case _:
                                # FIXME - maybe we should disallow non-named arguments and only allow kwargs in configs
                                # allows UDTs to be created via deferred instantiation
                                rv = self.create_factory(node, node.args, node.keywords, replace_with_instance=True)
                                return rv
                case _:
                    raise ConfigParseError(node, self.unparsed_input, f"configs do not support language element '{type_name(type(node))}'")

        except ConfigParseError:
            raise
        except Exception as e:
            raise ConfigParseError(node, self.unparsed_input, msg="Internal exception during configuration parsing " + str(e))
        #    raise ConfigParseError(node, self.unparsed_input, "") from e
        raise ConfigParseError(node, self.unparsed_input, msg = 'unsupported language element')
    
    def process_args_and_keywords(self, node_args, node_keywords):
        args = list([self.process(a) for a in node_args])

        kwargs = {}
        placeholders = {}
        for kw in node_keywords:
            if kw.arg is None:
                raise ConfigParseError(kw, self.unparsed_input, f"dictionary expansions '**' not allowed in config")
            if str(kw.arg) == 'self':
                raise ConfigParseError(kw, self.unparsed_input, f"may not pass self as an argument in configs")
            value = self.process(kw.value)

            # remove placeholder identifiers so they don't cause a collision
            if isinstance(value, LocalIdentifier):
                placeholders[kw.arg] = value
            else:
                kwargs[kw.arg] = value
        
        return args, kwargs, placeholders

    def create_factory(self, node, node_args, node_keywords, replace_with_instance:bool):
        func_node = node.func
        func_ident = self.process(func_node)
        if not isinstance(func_ident, IdentifierAccessor):
            raise ConfigParseError(func_node, self.unparsed_input, msg = f'huh got unexpected identifier {func_ident}')
        func_name = func_ident.fullid

        func_type = pydoc.locate(func_name)
        if func_type is None:
            raise ConfigParseError(node, self.unparsed_input, f"No such class or function found {func_name} while constructing Factory (are you missing the module name or import?)")
        if not callable(func_type) and not isinstance(func_type, type):
            raise ConfigParseError(node, self.unparsed_input, f"Factory requires a callable function or class but got: {func_name}")

        args, kwargs, placeholders = self.process_args_and_keywords(node_args=node_args, node_keywords=node_keywords)
        if func_name == "dataset.tokenizer.tokenize_join_and_slice":
            print("dataset.tokenizer.tokenize_join_and_slice", "args", args, "kwargs", kwargs, "placeholders", placeholders)

        try:
            rv = Factory(func_type, placeholders, *args, **kwargs)
            rv.replace_with_instance = replace_with_instance
        except Exception as e:
            raise ConfigParseError(node, self.unparsed_input, str(e)) from None
        return rv

def eval_first_expr(input : str):
    return ConfigParser().eval_first_expr(input)
