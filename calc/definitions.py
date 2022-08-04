from collections import namedtuple


class FunctionDefinition:
    precedence = 2
    associativity = 1

    def __init__(self, name, args, func, precedence=None, associativity=None):
        self.name = name
        self.args, self.f_args = self._get_args(args)
        self.func = func
        if precedence is not None:
            self.precedence = precedence
        if associativity is not None:
            self.associativity = associativity

    def is_func_arg(self, index):
        return index in self.f_args

    def __str__(self):
        return self.make_str(self.args)

    def make_str(self, inputs, bracketed=False):
        if inputs:
            if isinstance(self.func, CustomFunction):
                return str(self.func)
            return '{}({})'.format(self.name, ','.join(map(str, inputs)))
        else:
            return self.name

    def latex(self):
        return self.make_latex(self.args)

    def make_latex(self, inputs, bracketed=False):
        if inputs:
            if isinstance(self.func, CustomFunction):
                return _latex(self.func)
            return self.name + r'\left(' + ','.join(map(_latex, inputs)) + r'\right)'
        else:
            return self.name

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        return self.associativity == 1

    @staticmethod
    def _add_tex_brackets(s):
        return r'\left(' + s + r'\right)'

    @staticmethod
    def _get_args(args):
        args:list = list(args)
        f_args = dict()
        for i in range(len(args)):
            if isinstance(args[i], f_arg):
                f = args[i]
                args[i] = f.name
                f_args[i] = f.arg_count
        return args, f_args

class BinaryOperatorDefinition(FunctionDefinition):
    def __init__(self, name, func, precedence, associativity):
        super().__init__(name, ['a', 'b'], func, precedence, associativity)

    def make_str(self, inputs, bracketed=False):
        s = '{}{}{}'.format(inputs[0], self.name, inputs[1])
        if bracketed:
            s = '(' + s + ')'
        return s

    def make_latex(self, inputs, bracketed=False):
        s = _latex(inputs[0]) + self.name + _latex(inputs[1])
        if bracketed:
            s = self._add_tex_brackets(s)
        return s

    def __gt__(self, other):
        if self.precedence == other.precedence:
            return self.associativity == 1
        else:
            return self.precedence > other.precedence


class Constant:
    def __init__(self, name, value, display_name=None):
        self.name = name
        self.value = value
        # If display name is set, it will be used for str(self) instead of
        # the name. (used so constants can have multiple names, ex. 'π' and 'pi' -> 'π')
        self._dsply_name = display_name

    def __call__(self):
        return self.value

    def __str__(self):
        return self._dsply_name or self.name

    def __repr__(self):
        return str(self)

    def latex(self):
        return _latex_symbols.get(str(self), str(self))

class Variable:
    def __init__(self, ctx, name):
        self.ctx = ctx
        self.name = name

    def __call__(self):
        return self.ctx.get(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def latex(self):
        return str(self)

class Function:
    def __init__(self, ctx, definition, inputs, bracketed=False):
        if not isinstance(inputs, list):
            inputs = OutputList(inputs)
        self.definition = definition
        self.inputs = inputs
        self.bracketed = bracketed
        self.ctx = ctx
        self._verify_inputs()

    def set_arg(self, arg, val):
        index = self.definition.args.index(arg)
        self.inputs[index] = val

    def __call__(self):
        inputs = self._eval_inputs()
        return self.definition.func(*inputs)

    def _eval_inputs(self):
        for item in self.inputs:
            yield _eval_item(item)

    def _verify_inputs(self):
        if len(self.inputs) != len(self.definition.args):
            raise SyntaxError(
                "{} expected {} argument(s), but {} were given."
                .format(self.definition, len(self.definition.args), len(self.inputs))
            )

        for (i, arg_count) in self.definition.f_args.items():
            if not isinstance(self.inputs[i], _eval_time_funcs):
                raise SyntaxError(
                    "Argument '{}' of {} must be a function, not {}."
                    .format(self.definition.args[i], self.definition, self.inputs[i])
                )

            in_arg_count = len(self.inputs[i].definition.args)
            if in_arg_count != arg_count:
                raise SyntaxError(
                    "Argument '{}' of {} expected a function that takes {} argument(s), but {} takes {}."
                    .format(self.definition.args[i], self.definition, arg_count, self.inputs[i].definition, in_arg_count)
                )

    def __str__(self):
        return self.definition.make_str(self.inputs, self.bracketed)

    def __repr__(self):
        return str(self)

    def latex(self):
        return self.definition.make_latex(self.inputs, self.bracketed)

class OutputList(list):
    def __init__(self, *items):
        super().__init__(items)

    def __call__(self):
        return [_eval_item(item) for item in self]

    def latex(self):
        return r'\left[' + ','.join(map(_latex, self)) + r'\right]'

    @staticmethod
    def combine(*items):
        result = OutputList()
        for item in items:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

class CustomFunction:
    def __init__(self, ctx, definition):
        self.definition = definition
        self.ctx = ctx

    def __call__(self, *inputs):
        if len(inputs) != len(self.definition.args):
            raise SyntaxError("{} expected {} argument(s), but {} were given."
                              .format(self.definition, len(self.definition.args), len(inputs)))
        with self.ctx.with_context():
            for arg, val in zip(self.definition.args, inputs):
                self.ctx.set(arg, val)
            if isinstance(self.definition.func, _parse_time_funcs):
                return self.definition.func()
            else:
                return self.definition.func

    def __str__(self):
        return '{} = {}'.format(self.definition, self.definition.func)

    def __repr__(self):
        return str(self)

    def latex(self):
        return _latex(self.definition) + '=' + _latex(self.definition.func)

class FunctionArgument:
    def __init__(self, definition):
        self.definition = definition

    def __call__(self, *inputs):
        if len(inputs) != len(self.definition.args):
            raise SyntaxError("{} expected {} argument(s), but {} were given."
                              .format(self.definition, len(self.definition.args), len(inputs)))
        return self.definition.func(*inputs)

    def __str__(self):
        return str(self.definition)

    def __repr__(self):
        return str(self)

    def latex(self):
        if isinstance(self.definition.func, CustomFunction):
            return _latex(self.definition.func.definition.func)
        else:
            return _latex(self.definition)

f_arg = namedtuple('f_arg', 'name arg_count')

# Represents function-like classes that have inputs and such supplied
# at parse-time, and use __call__ with no arguments to evaluate.
_parse_time_funcs = (Constant, Variable, OutputList, Function)
# Represents function-like classes that have inputs and such not
# defined until evaluation-time, and use __call__ to supply inputs
# and evaluate.
_eval_time_funcs = (CustomFunction, FunctionArgument)


def _eval_item(item):
    if isinstance(item, _parse_time_funcs):
        return item()
    else:
        return item

def _latex(item, braces=True):
    if isinstance(item, (int, float)):
        item = str(item)
    elif isinstance(item, str):
        item = _latex_symbols.get(item, item)
    else:
        item = item.latex()
    if braces:
        item = '{' + item + '}'
    return item

_latex_symbols = {
    'π': r'\pi',
    'θ': r'\theta',
    'ϕ': r'\phi',
    '∞': r'\infty',
}
