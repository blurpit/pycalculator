from collections import namedtuple

import numpy as np


class FunctionDefinition:
    precedence = 2
    associativity = 1

    def __init__(self, name, args, func, precedence=None, associativity=None, disable_arg_count_check=False):
        self.name = name
        self.args, self.f_args = self._get_args(args)
        self.func = func
        if precedence is not None:
            self.precedence = precedence
        if associativity is not None:
            self.associativity = associativity
        self.disable_arg_count_check = disable_arg_count_check

    def is_func_arg(self, index):
        return index in self.f_args

    @property
    def signature(self):
        return '{}({})'.format(self.name, ', '.join(map(str, self.args)))

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

    @property
    def signature(self):
        return str(self)

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
            inputs = Vector([inputs])
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
        if not self.definition.disable_arg_count_check and len(self.inputs) != len(self.definition.args):
            raise SyntaxError(
                "{} expected {} argument(s), but {} were given."
                .format(self.definition.signature, len(self.definition.args), len(self.inputs))
            )

        for (i, arg_count) in self.definition.f_args.items():
            if not isinstance(self.inputs[i], _eval_time_funcs):
                raise SyntaxError(
                    "Argument '{}' of {} must be a function, not {}."
                    .format(self.definition.args[i], self.definition.signature, self.inputs[i])
                )

            in_arg_count = len(self.inputs[i].definition.args)
            if in_arg_count != arg_count:
                raise SyntaxError(
                    "Argument '{}' of {} expected a function that takes {} argument(s), but {} takes {}."
                    .format(self.definition.args[i], self.definition.signature, arg_count, self.inputs[i].definition.signature, in_arg_count)
                )

    def __str__(self):
        return self.definition.make_str(self.inputs, self.bracketed)

    def __repr__(self):
        return str(self)

    def latex(self):
        return self.definition.make_latex(self.inputs, self.bracketed)

class Vector(list):
    def __init__(self, items=None, vertical=True):
        if items is None:
            items = []
        super().__init__(items)
        self.bracketed = False
        self.vertical = vertical

    def __call__(self):
        return Vector(_eval_item(item) for item in self)

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise TypeError("unsupported operand type(s) for +: 'Vector' and '{}'".format(type(other).__name__))
        if len(other) != len(self):
            raise ValueError("Vector addition undefined for vectors of length {} and {}".format(len(self), len(other)))
        return Vector(a + b for a, b in zip(self, other))

    def __mul__(self, other):
        if not isinstance(other, (float, int)):
            raise TypeError("unsupported operand type(s) for *: 'Vector' and '{}'".format(type(other).__name__))
        return Vector(other * item for item in self)

    __radd__ = __add__
    __rmul__ = __mul__

    def latex(self):
        if self.vertical:
            return r'\begin{bmatrix}' + r'\\'.join(map(_latex, self)) + r'\end{bmatrix}'
        else:
            return r'\left[' + ','.join(map(_latex, self)) + r'\right]'

    def __str__(self):
        return '(' + ', '.join(map(str, self)) + ')'

    def __repr__(self):
        return str(self)

    @staticmethod
    def combine(a, b):
        result = Vector()
        if isinstance(a, Vector) and not a.bracketed: result.extend(a)
        else: result.append(a)
        if isinstance(b, Vector) and not b.bracketed: result.extend(b)
        else: result.append(b)
        return result

class Matrix(list):
    def __init__(self, rows):
        super().__init__(rows)
        self.shape = self._verify()

    def __call__(self):
        return Matrix(col() for col in self)

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("unsupported operand type(s) for +: 'Matrix' and '{}'".format(type(other).__name__))
        if self.shape != other.shape:
            raise ValueError('Incompatible matrix addition shapes: {} and {}'.format(self.shape, other.shape))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self[i][j] += other[i][j]
        return self

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self[i][j] *= other
            return self
        elif isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError('Incompatible matrix multiplication shapes: {} and {}'.format(self.shape, other.shape))
            return Matrix(
                Vector(row) for row in np.matmul(self, other)
            )
        else:
            raise TypeError("unsupported operand type(s) for *: 'Matrix' and '{}'".format(type(other).__name__))

    __radd__ = __add__

    def latex(self):
        get_row = lambda r: '&'.join(map(_latex, r))
        return r'\begin{bmatrix}' + r'\\'.join(map(get_row, self)) + r'\end{bmatrix}'

    def __str__(self):
        return 'mat(' + ', '.join(map(str, self)) + ')'

    def __repr__(self):
        return str(self)

    def _verify(self):
        if len(self) < 1:
            return 0, 0
        for i, row in enumerate(self): # convert singletons to vectors
            if not isinstance(row, Vector):
                self[i] = Vector([row])
        cols = len(self[0])
        if any(len(row) != cols for row in self):
            raise ValueError('Matrix row vectors must be the same length')
        return len(self), cols


class CustomFunction:
    def __init__(self, ctx, definition):
        self.definition = definition
        self.ctx = ctx

    def __call__(self, *inputs):
        if len(inputs) != len(self.definition.args):
            raise SyntaxError("{} expected {} argument(s), but {} were given."
                              .format(self.definition.signature, len(self.definition.args), len(inputs)))
        with self.ctx.with_context():
            for arg, val in zip(self.definition.args, inputs):
                self.ctx.set(arg, val)
            if isinstance(self.definition.func, _parse_time_funcs):
                return self.definition.func()
            else:
                return self.definition.func

    def __str__(self):
        return '{} = {}'.format(self.definition.signature, self.definition.func)

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
                              .format(self.definition.signature, len(self.definition.args), len(inputs)))
        return self.definition.func(*inputs)

    def __str__(self):
        return str(self.definition)

    def __repr__(self):
        return str(self)

    def latex(self):
        return _latex(self.definition)

f_arg = namedtuple('f_arg', 'name arg_count')

# Represents function-like classes that have inputs and such supplied
# at parse-time, and use __call__ with no arguments to evaluate.
_parse_time_funcs = (Constant, Variable, Function, Vector, Matrix)
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
        item = round(item, 6)
        if item % 1 == 0:
            item = int(item)
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
