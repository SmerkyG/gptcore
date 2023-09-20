from pydoc import locate

import typing
import inspect
import collections

# def instantiate(required_type : type, args : dict, **kwargs):
#     if 'class' in args:
#         if(args['class'] is None):
#             return None
#         subclass = locate(args['class'])
#         if subclass is None:
#             raise Exception(f"no matching class found for `{args['class']}` (did you forget to include the module name?)")
#         if not issubclass(subclass, required_type):
#             raise Exception(f'{subclass} must be a subclass of {required_type}')
#         required_type = subclass
#         args = dict(args)
#         del args['class']
#         sig = inspect.signature(subclass.__init__)
#         for key, value in args.items():
#             if isinstance(value, dict):
#                 ann = sig.parameters.get(key).annotation
#                 subtype = ann#typing.Any#type(ann)
#                 args[key] = instantiate(subtype, value)
#     args_merged = dict(args)
#     args_merged.update(kwargs)
#     return required_type(**args_merged)

def type_name(t):
    if type(t) != type:
        return str(t)
    origin = typing.get_origin(t)
    if origin is None:
        return ((t.__module__ + '.') if t.__module__ != 'builtins' else '') + t.__name__
    else:
        rv = type_name(origin) + '['
        for arg in typing.get_args(t):
            rv += type_name(arg)
        rv += ']'
        return rv

# just like is_instance but with support for generics, including testing values against Unions of Literals!
def is_generic_instance(a, b):
    if b == typing.Any:
        return True
    if b == float:
        # allow narrowing from float to int
        return isinstance(a, (float, int))
    match b_origin := typing.get_origin(b):
        #case typing.Optional: # seems that we don't need this case bc Python implements Optional as Union[T, type(None)]
        #    return is_generic_instance(a, typing.get_args(b)[0])
        case typing.Union:
            for arg in typing.get_args(b):
                if is_generic_instance(a, arg):
                    return True
            return False
        case typing.Literal:
            for arg in typing.get_args(b):
                if a == arg:
                    return True
            return False
        case typing.Iterable | collections.abc.Iterable:
            b_subtype = typing.get_args(b)[0]
            # NOTE - relaxed constraint here, if a is Iterable with no generic parameters we allow it through even if b has specific generic Iterable requirements
            # NOTE - this isn't as good as the commented check below, but unfortunately Factory won't have instantiated its class yet, so we would end up checking Factory == subtype, which would fail
            if isinstance(a, typing.Iterable) and len(typing.get_args(a)) == 0:
                return True
            # if isinstance(a, typing.Iterable) and len(typing.get_args(a)) == 0:
            #     # a is an Iterable with no generic parameters, so iterate it and check that every element matches b_subtype!
            #     for v in a:
            #         if not is_generic_instance(v, b_subtype):
            #             print(f"iterable subtype mismatch {v} {b_subtype}")
            #             return False
            #     return True                
            return is_generic_instance(a, b_subtype)
        case None:
            return isinstance(a, b)
        case _:
            # found an unrecognized generic type (likely a UDT, unless we missed a case)
            return isinstance(a, b_origin)
            raise Exception(f"reached unimplemented case of is_generic_instance: {a} {b} {typing.get_origin(b)}")

