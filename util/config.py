import typing
import types
import shlex
import inspect
import types
import collections.abc
from collections import OrderedDict

from util.type_utils import type_name, is_generic_instance

from util.locate import locate, Missing

T = typing.TypeVar("T")
class IPartial(typing.Generic[T]):
    def __init__(self, immediate:bool):
        self.immediate = immediate

def recursively_replace_immediate_ipartials_as_needed(c):
    if isinstance(c, list):
        for i, v in enumerate(c):
            c[i] = recursively_replace_immediate_ipartials_as_needed(v)
    elif isinstance(c, dict):
        for k, v in c.items():
            c[k] = recursively_replace_immediate_ipartials_as_needed(v)
    elif isinstance(c, Factory) and c.immediate:
        c = c()
    elif isinstance(c, MemberAccessor) and c.immediate:
        c = c()
    elif isinstance(c, IdentifierAccessor) and c.immediate:
        c = c()
    return c

def recursively_replace_identifier_accessors(c):
    match c:
        case list():
            for i, v in enumerate(c):
                c[i] = recursively_replace_identifier_accessors(v)
        case dict():
            for k, v in c.items():
                c[k] = recursively_replace_identifier_accessors(v)
        case Factory():
            for i, v in enumerate(c.args):
                c.args[i] = recursively_replace_identifier_accessors(v)
            for k, v in c.kwargs.items():
                c.kwargs[k] = recursively_replace_identifier_accessors(v)
        case MemberAccessor():
            c.inner_ipartial = recursively_replace_identifier_accessors(c.inner_ipartial)
            for i, v in enumerate(c.args):
                c.args[i] = recursively_replace_identifier_accessors(v)
            for k, v in c.kwargs.items():
                c.kwargs[k] = recursively_replace_identifier_accessors(v)
        case IdentifierAccessor():
            #c_old = c
            c = c()
            #print(f"Replaced IdentifierAccessor {c_old} with {c}")
    return c
    
class Factory(typing.Generic[T], IPartial[T]):
    def __init__(self, fn : typing.Union[type, typing.Callable, str, None] = None, *args, **kwargs):
        super().__init__(False)
        self.placeholders = {}
        self.positional_placeholders_count = 0

        if isinstance(fn, str):
            self.fn = locate(fn)
            if self.fn is None:
                raise Exception(f"No such class or function found {fn} during Factory.init (are you missing the module name or import?)")
        else:
            self.fn = fn

        if self.fn is not None and not callable(self.fn) and not isinstance(self.fn, type):
            raise Exception(f"Factory requires a callable function or class but got: {self.fn}")

        self.args = list(args)
        self.kwargs = dict(kwargs)

    def __call__(self, /, *args, **kwargs) -> T:
        if self.fn is None:
            return None

        args = [*self.args, *args]

        # FIXME - is this the kind of merge we want?
        kwargs = {**self.kwargs, **kwargs}
        #kwargs = dict(self.kwargs)
        #kwargs.update(kwargs)

        for i, v in enumerate(args):
            args[i] = recursively_replace_immediate_ipartials_as_needed(v)
        for k, v in kwargs.items():
            kwargs[k] = recursively_replace_immediate_ipartials_as_needed(v)

        return self.fn(*args, **kwargs)
    
    def toDict(self):
        if self is None:
            return None
        def _process(node):
            match node:
                case Factory():
                    rv = {"class_path":type_name(node.fn)}
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
        rv = f"{type(self).__name__}({type_name(self.fn)}"
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
    def __init__(self, fullid : str, immediate:bool = True):
        super().__init__(immediate)
        #print(f"IdentifierAccessor {fullid}")
        self.fullid = fullid

    def __call__(self) -> typing.Any: # can't know the return type, unfortunately, since it's a string based identifier accessor
        rv = locate(self.fullid)
        #print(f"Calling IdentifierAccessor {self.fullid} == {rv}")
        return rv

    def __repr__(self):
        return f"{type(self).__name__}({self.fullid})"

class MemberAccessor(typing.Generic[T], IPartial[T]):
    def __init__(self, member_name : str, inner_ipartial : IPartial[T], is_call:bool = False, immediate:bool = True, positional_placeholders_count=0, placeholders = {}, *args, **kwargs):
        super().__init__(immediate)
        self.positional_placeholders_count = positional_placeholders_count
        self.placeholders = placeholders

        self.inner_ipartial = inner_ipartial
        self.member_name = member_name
        self.is_call = is_call
        self.args = list(args)
        self.kwargs = dict(kwargs)

    def __call__(self, /, *args, **kwargs) -> typing.Any: # can't know the return type, unfortunately, since it's a string based property accessor or member fn invocation
        #print(f"MEMBERACCESSOR {self.inner_ipartial}", self.is_call, args, self.args, kwargs, self.kwargs)

        args = [*self.args, *args]

        # FIXME - is this the kind of merge we want?
        kwargs = {**self.kwargs, **kwargs}
        kwargs = dict(self.kwargs)
        kwargs.update(kwargs)

        #for i, v in enumerate(args):
        #    args[i] = recursively_immediate_ipartials_as_needed(v)
        #for k, v in kwargs.items():
        #    kwargs[k] = recursively_immediate_ipartials_as_needed(v)

        obj = recursively_replace_immediate_ipartials_as_needed(self.inner_ipartial)

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

# def ImmediateFactory(fn : type | Callable | str | None = None, *args, **kwargs):
#     rv = Factory(fn, *args, **kwargs)
#     rv.immediate = True
#     return rv

# def merge(dst : Factory | dict, src : Factory | dict):
#     # merge instances of Factory, dict, and list
#     if type(src) == type(dst):
#         if isinstance(src, Factory):
#             # FIXME - add some sort of required superclass too
#             if src.fn != type(None):
#                 dst.fn = src.fn
#             #dst.args = src.args
#             dst.kwargs = merge(dst.kwargs, src.kwargs)
#             return dst
#         elif isinstance(src, dict):
#             for k, srcv in src.items():
#                 dst[k] = merge(dst[k], srcv) if k in dst else srcv
#             return dst
#         elif isinstance(src, list):
#             dst.clear()
#             for i, srcv in enumerate(src):
#                 dst[i] = merge(dst[i], srcv)
#             return dst
#     # otherwise, just copy it
#     return src

def typecheck(path : str, obj : typing.Any, required_type : type = typing.Any, is_immediate_factory:bool = True):
    errors = ''
    try:
        # print(f"typecheck {path} {required_type}")
        if isinstance(required_type, str):
            # if the required type is a string (which could happen due to weird python pre-declarations that aren't available, and is done in lightning.LightningModule.fit's model parameter type)
            # then just allow anything through, since we can't realistically type check this
            required_type = typing.Any

        #if type == typing.Any:
        #    return errors

        if isinstance(obj, IdentifierAccessor):
            # allow IdentifierAccessors through, since they return Any
            return errors


        if not is_generic_instance(obj, required_type):           
            if not isinstance(obj, Factory):
                #    print(f"Factory found {v.fn}")
                # get_config_func_options(parent_key, value) + \
                errors += f"Config Type Mismatch: expected {type_name(required_type)} but got {type_name(type(obj))}\n in config setting `{path}` : {type_name(required_type)} = {obj}\n"
                return errors

        if isinstance(obj, Factory):
            if typing.get_origin(required_type) == collections.abc.Callable and obj.immediate: # wanted a factory but got what is probably a method invocation
                errors += f"Config Type Mismatch: expected Factory/Callable but got immediate invocation of {type_name(obj.fn)} (did you forget to prefix this with 'lambda:'?)\n in config setting `{path}` : {type_name(required_type)} = {obj}\n"
                return errors

            # fill in fn when Empty but we have a type annotation
            # FIXME - add and check some sort of required superclass too
            if obj.fn == type(None):
                obj.fn = required_type

            if isinstance(obj.fn, types.FunctionType):
                sig = inspect.signature(obj.fn)
                is_init_fn = False
            else:
                sig = inspect.signature(obj.fn.__init__)
                is_init_fn = True

            # check for missing arguments only if there's no dictionary expansion present
            if "**" not in obj.placeholders:
                arg_index = -1
                for k, p in sig.parameters.items():
                    # skip check for 'self' for member functions
                    if k == 'self':
                        continue

                    arg_index += 1

                    if arg_index < obj.positional_placeholders_count:
                        continue

                    #if kind in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD):
                    if k in obj.kwargs or k in obj.placeholders:
                        continue

                    if p.default is not p.empty:
                        continue

                    if p.annotation is p.empty:
                        continue

                    if (p.annotation != typing.Any and typing.get_origin(p.annotation) != typing.Optional):
                        if not (typing.get_origin(p.annotation) in [typing.Union, types.UnionType] and len(typing.get_args(p.annotation)) == 2 and typing.get_args(p.annotation)[1]==type(None)):
                            return f"Missing config setting `{path}.{k}` : {type_name(p.annotation)}\n"

            # traverse all subelements recursively
            for k, v in obj.kwargs.items():
                if k not in sig.parameters.keys():
                    return f'Disallowed config entry `{path}.{k}` - No such parameter {k} in {obj.fn}\n'
                p = sig.parameters[k]
                rt = p.annotation
                if rt == inspect.Parameter.empty:
                    rt = typing.Any
                errors += typecheck(path + '.' + k, v, rt, obj.immediate)


        elif isinstance(obj, dict):
            # traverse all subelements recursively
            for k, v in obj.items():
                errors += typecheck(path + '.' + k, v)
        elif isinstance(obj, list):
            # traverse all subelements recursively
            for i, v2 in enumerate(obj):
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
        else:
            msg += " [NO LINE NUMBER AVAILABLE]"
        super().__init__(msg)

class LocalIdentifier():
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return f"LocalIdentifier('{self.name}')"

class ConfigParser():
    def eval_first_expr(self, unparsed_input : str, incoming_macros : dict):
        self.locals = OrderedDict()
        self.macros = OrderedDict()
        self.unparsed_input = unparsed_input
        self.imports_map = {}
        root = ast.parse(unparsed_input)
        if isinstance(root, Module):
            # get the first expression in the module
            nodes = root.body
            for i, node in enumerate(nodes):
                if isinstance(node, Import):
                    for alias in node.names:
                        if alias.asname is not None:
                            self.imports_map[alias.asname] = alias.name
                elif isinstance(node, ImportFrom):
                    for alias in node.names:
                        self.imports_map[alias.asname if alias.asname is not None else alias.name] = '.'.join([str(node.module), alias.name])
                elif isinstance(node, Assign):
                    # macro style assignments in global module scope
                    if len(node.targets) != 1 or not isinstance(node.targets[0], Name):
                        raise ConfigParseError(root, self.unparsed_input, "unsupported language element: multiple or complex assignment")
                    id = node.targets[0].id
                    if id in incoming_macros:
                        parsed_macro = ast.parse(incoming_macros[id])
                        if not isinstance(parsed_macro, Module) or len(parsed_macro.body) == 0:
                            raise ConfigParseError(parsed_macro, incoming_macros[id], f"error parsing - commandline macro value for '{id}'")
                        if not isinstance(parsed_macro.body[0], Expr):
                            raise ConfigParseError(parsed_macro.body, incoming_macros[id], f"error parsing - commandline macro value for '{id}' was not an expression")
                        self.macros[id] = parsed_macro.body[0].value
                    else:
                        self.macros[id] = node.value
                elif isinstance(node, Expr):
                    if i != len(nodes)-1:
                        raise ConfigParseError(nodes[i+1], self.unparsed_input, "configuration must not contain more elements after first top level expression")
                    for key in incoming_macros.keys():
                        if key not in self.macros:
                            raise ConfigParseError(root, self.unparsed_input, f"'{key}' not found in config globals\nglobal values set via commandline must override existing globals set in the config")
                    return self.process(node.value)
                else:
                    raise ConfigParseError(node, self.unparsed_input, f"configs do not support module-level elements of type '{type_name(type(node))}'")
                    
                
        raise ConfigParseError(root, self.unparsed_input, "configuration did not contain a top level expression")

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
                        return MemberAccessor(member_name=node.attr, inner_ipartial=value, is_call=False, immediate=True, placeholders={})
                        #raise ConfigParseError(node, self.unparsed_input, 'configuration files do not support member access (neither properties nor functions)')
                    id = value.fullid + '.' + str(node.attr)
                    fullid = self.imports_map[id] if id in self.imports_map else id

                    located = locate(fullid, Missing)
                    if located is Missing:
                        raise ConfigParseError(node, self.unparsed_input, f"could not locate identifier '{id}'")
                    return IdentifierAccessor(fullid)
                case Name():
                    id = node.id

                    # check if it's a macro
                    if id in self.macros:
                        return self.process(self.macros[id])

                    # check if it's a local
                    if id in self.locals:
                        # it's a local identifier
                        return LocalIdentifier(id)

                    fullid = self.imports_map[id] if id in self.imports_map else id
                    return IdentifierAccessor(fullid)
                case Lambda():
                    locals_tmp = self.locals
                    # FIXME - this next commented line disallows lambda args
                    # if len(node.args.posonlyargs) > 0 or len(node.args.args) > 0 or len(node.args.kwonlyargs) > 0:                    
                    if len(node.args.args) > 0:
                        self.locals = self.locals.copy()
                        for arg in node.args.args: self.locals[arg.arg] = True#LocalIdentifier(arg.arg)

                    # allows dictionary expansion args like **kwarg
                    if node.args.kwarg is not None:
                        self.locals = self.locals.copy()
                        self.locals[node.args.kwarg.arg] = True#LocalIdentifier(node.args.kwarg.arg)

                    if not isinstance(node.body, Call) or not isinstance(node.body.func, (Name, Attribute)):
                        raise ConfigParseError(node, self.unparsed_input, 'configuration lambda must be used to return a Factory or MemberAccessor')
                        #raise ConfigParseError(node, self.unparsed_input, 'configuration lambda must have zero arguments, and is used to return a Factory')
                    # FIXME - instead of calling create_factory directly here we could somehow indicate that the result should be immediate=False and just always process node.body.func
                    node = node.body
                    if isinstance(node.func, Attribute) and isinstance(node.func.value, Call):
                        # create MemberAccessor
                        positional_placeholders_count, placeholders, args, kwargs = self.process_args_and_keywords(node_args=node.args, node_keywords=node.keywords)
                        #print("CAPTURING MemberAccess ", node.func.attr, placeholders, args, kwargs)
                        return MemberAccessor(member_name=node.func.attr, inner_ipartial=self.process(node.func.value), is_call=True, immediate=False, positional_placeholders_count=positional_placeholders_count, placeholders=placeholders, *args, **kwargs)
                    else:
                        # create Factory
                        rv = self.create_factory(node, node.args, node.keywords, immediate=False)

                    self.locals = locals_tmp
                    return rv
                case Call():
                    if isinstance(node.func, Attribute) and isinstance(node.func.value, Call):
                        #print()
                        #print(ast.dump(node))
                        #print()
                        # create MemberAccessor
                        positional_placeholders_count, placeholders, args, kwargs = self.process_args_and_keywords(node_args=node.args, node_keywords=node.keywords)
                        #print("CAPTURING CallMemberAccess ", node.func.attr, placeholders, args, kwargs)
                        return MemberAccessor(member_name=node.func.attr, inner_ipartial=self.process(node.func.value), is_call=True, immediate=True, positional_placeholders_count=positional_placeholders_count, placeholders=placeholders, *args, **kwargs)

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
                            case 'util.config.Factory':
                                return self.create_factory(node.args[0], node.args[1:], node.keywords, immediate=False)
                            case _:
                                # FIXME - maybe we should disallow non-named arguments and only allow kwargs in configs
                                # allows UDTs to be created via deferred instantiation
                                rv = self.create_factory(node, node.args, node.keywords, immediate=True)
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
        positional_placeholders_count = 0
        args = []
        locals_list = None
        for i, value in enumerate(map(self.process, node_args)):
            if isinstance(value, LocalIdentifier):
                if locals_list is None:
                    locals_list = list(self.locals.keys())
                if len(self.locals) <= i:
                    raise ConfigParseError(node_args[i], self.unparsed_input, f"in configs, lambda local identifiers may only be passed to positional arguments in the order they came into the lambda. '{value.name}' came after all locals were exhausted")
                elif value.name != locals_list[i]:
                    raise ConfigParseError(node_args[i], self.unparsed_input, f"in configs, lambda local identifiers may only be passed to positional arguments in the order they came into the lambda. '{value.name}' should be '{locals_list[i]}'")
                positional_placeholders_count += 1
            else:
                args.append(value)

        kwargs = {}
        placeholders = {}
        for kw in node_keywords:
            #if kw.arg is None:
                #print(ast.dump(kw.value))
                #raise ConfigParseError(kw, self.unparsed_input, f"dictionary expansions '**' not allowed in config")
            if str(kw.arg) == 'self':
                raise ConfigParseError(kw, self.unparsed_input, f"may not pass self as an argument in configs")
            value = self.process(kw.value)

            # remove placeholder identifiers so they don't cause a collision
            if isinstance(value, LocalIdentifier):
                if kw.arg is None:
                    # was a dictonary expansion, so add special value into placeholders to signify that
                    placeholders["**"] = "**"
                else:
                    # for now, disallow remapping of names from lambda input args to factory kwargs
                    if value.name != str(kw.arg):
                        raise ConfigParseError(kw, self.unparsed_input, f"in configs, lambda local identifiers may only be passed to keyword arguments of the same name. Found mismatched: {kw.arg}={value.name}")
                    placeholders[kw.arg] = value
            else:
                kwargs[kw.arg] = value
        
        return positional_placeholders_count, placeholders, args, kwargs

    def create_factory(self, node, node_args, node_keywords, immediate:bool):
        func_node = node.func
        func_ident = self.process(func_node)
        if not isinstance(func_ident, IdentifierAccessor):
            raise ConfigParseError(func_node, self.unparsed_input, msg = f'huh got unexpected identifier {func_ident}')
        func_name = func_ident.fullid

        fn = locate(func_name)
        if fn is None:
            raise ConfigParseError(node, self.unparsed_input, f"No such class or function found {func_name} while constructing Factory (are you missing the module name or import?)")
        if not callable(fn) and not isinstance(fn, type):
            raise ConfigParseError(node, self.unparsed_input, f"Factory requires a callable function or class but got: {func_name}")

        positional_placeholders_count, placeholders, args, kwargs = self.process_args_and_keywords(node_args=node_args, node_keywords=node_keywords)

        try:
            rv = Factory(fn, *args, **kwargs)
            rv.immediate = immediate
            rv.placeholders = placeholders
            rv.positional_placeholders_count = positional_placeholders_count
        except Exception as e:
            raise ConfigParseError(node, self.unparsed_input, str(e)) from None
        return rv

def eval_first_expr(unparsed_input : str, incoming_macros : dict):
    return ConfigParser().eval_first_expr(unparsed_input, incoming_macros)
