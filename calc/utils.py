import json
import math
import operator
import textwrap
from io import BytesIO
from random import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from latex import build_pdf
from pdf2image import convert_from_bytes
from scipy.fftpack import fft
from scipy.integrate import quad
from scipy.misc import derivative

from .context import Context
from .definitions import FunctionDefinition, BinaryOperatorDefinition, Constant, Function, CustomFunction, f_arg, \
    FunctionArgument, _parse_time_funcs, _eval_time_funcs, _latex, _latex_symbols
from .parser import parse
from .tokenizer import tokenize

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rc('font', family='serif')
plt.rc('text', color='white')
plt.rc('mathtext', fontset='dejavuserif')
plt.rc('axes', facecolor='none', edgecolor='none',
       labelsize=28, titlesize=32, labelcolor='white',
       axisbelow=True, grid=True)
plt.rc('grid', color='#202225', linestyle='solid', lw=3)
plt.rc('xtick', direction='out', labelsize=18, color='#dcddde')
plt.rc('ytick', direction='out', labelsize=18, color='#dcddde')
plt.rc('lines', linewidth=5)
plt.rc('figure', facecolor='#37393f', figsize=[12, 10], dpi=72)


def evaluate(ctx:Context, expression:str):
    expression = expression.replace(' ', '')
    tokens = tokenize(ctx, expression)
    parsed = parse(ctx, tokens)
    if isinstance(parsed, _parse_time_funcs):
        result = parsed()
    else:
        result = parsed
    ctx.ans = result
    return result

def console(ctx:Context):
    bold = '\033[01m'
    red = '\033[31m'
    yellow = '\033[93m'
    reset = '\033[0m'

    def cprint(s, col):
        print(col + str(s) + reset)

    result = None
    ctx.push_context()
    while True:
        exp = input(yellow + '>>> ' + reset)
        if exp == 'exit':
            break
        elif exp == 'ctx':
            print(ctx)
        elif exp == 'resetctx':
            ctx.pop_context()
            ctx.push_context()
        elif exp == 'graph':
            fig = graph(ctx, result, -10, 10, -10, 10)
            fig.show()
        else:
            try:
                result = evaluate(ctx, exp)
                if isinstance(result, CustomFunction):
                    ctx.add(result)
                cprint(result, bold)
            except Exception as e:
                cprint('{}: {}'.format(e.__class__.__name__, str(e)), red)
    ctx.pop_context()

def graph(ctx:Context, func, xlow=-10, xhigh=10, ylow=None, yhigh=None, n=1000, tex_title=True):
    if isinstance(func, str):
        func = func.replace(' ', '')
        if func in ctx:
            func = ctx.get(func)
        else:
            expression = func.removeprefix('y=')
            tokens = tokenize(ctx, expression)
            func = parse(ctx, tokens)
    if isinstance(func, FunctionDefinition):
        func = FunctionArgument(func)
    elif isinstance(func, _parse_time_funcs):
        definition = FunctionDefinition('f', 'x', func)
        func = CustomFunction(ctx, definition)
    elif not isinstance(func, _eval_time_funcs):
        raise TypeError("Cannot graph type '{}', must be a function.".format(func.__class__.__name__))
    if len(func.definition.args) != 1:
        raise TypeError("{} is not 1-dimensional (function must take 1 input and return 1 output).".format(func.definition.signature))

    x = np.linspace(xlow, xhigh, n)
    y = np.empty(len(x))

    for i in range(len(x)):
        result = func(x[i])
        if not isinstance(result, (float, int)):
            raise TypeError("{} is not 1-dimensional (function must take 1 input and return 1 output).".format(func.definition.signature))
        y[i] = float(result)

    fig, ax = plt.subplots(1, 1)

    ax.axhline(0, color='#202225', lw=6)
    ax.axvline(0, color='#202225', lw=6)

    ax.set_xlim(xlow, xhigh)
    if ylow is not None and yhigh is not None:
        ax.set_ylim(ylow, yhigh)
    ax.set_xlabel(str(func.definition.args[0]))
    ax.set_ylabel(str(func.definition.signature))
    if tex_title:
        ax.set_title('${}$'.format(func.latex()))
    else:
        ax.set_title(textwrap.fill(str(func), 48))

    ax.plot(x, y, color='#ed4245')
    return fig

def savefig_bytesio(fig):
    bio = BytesIO()
    fig.savefig(bio, format='png', bbox_inches='tight')
    plt.close(fig)
    bio.seek(0)
    return bio

def savefig_png(fig, path):
    fig.savefig(path, format='png', bbox_inches='tight')

def latex(ctx, expression):
    if isinstance(expression, str):
        expression = expression.replace(' ', '')
        if expression in ctx:
            parsed = ctx.get(expression)
        else:
            parsed = parse(ctx, tokenize(ctx, expression))
    else:
        parsed = expression
    return _latex(parsed, braces=False)

def latex_to_image(tex, dpi=200, color='white', transparent=True):
    tex_template = r"""
    \def\formula{%s}
    \documentclass[border=2pt]{standalone}
    \usepackage{amsmath}
    \usepackage{varwidth}
    \usepackage{xcolor}
    \begin{document}
    \color{%s}
    \begin{varwidth}{\linewidth}
    \[ \formula \]
    \end{varwidth}
    \end{document}
    """
    pdf = build_pdf(tex_template % (tex, color))
    images = convert_from_bytes(
        bytes(pdf),
        dpi=dpi,
        fmt='png',
        single_file=True,
        transparent=transparent
    )
    return images[0]

def load_contexts(ctx, fp):
    """
    Load a context stack from a json file into an existing context object
    :param ctx: Starting context
    :param fp: Filepath
    """
    with open(fp, 'r') as file:
        contexts = json.load(file)
    for element in contexts:
        ctx.push_context()
        for name, expression in element.items():
            expression = expression.replace(' ', '')
            tokens = tokenize(ctx, expression)
            parsed = parse(ctx, tokens)
            if isinstance(parsed, (float, int)):
                ctx.set(name, parsed)
            else:
                ctx.add(parsed)

def save_contexts(ctx, fp):
    """
    Save a context stack into a json file. Does not include the context stack root.
    :param ctx: Context to save
    :param fp: Filepath
    """
    result = []
    for element in ctx.ctx_stack[1:]:
        result.append({})
        for name, item in element.items():
            if isinstance(item, FunctionDefinition):
                item = item.func
            result[-1][name] = str(item)
    with open(fp, 'w') as file:
        json.dump(result, file)

def is_identifier(name):
    return name.isalpha()


class NegationDefinition(FunctionDefinition):
    precedence = 4
    def make_str(self, inputs, bracketed=False):
        return '-' + str(inputs[0])

    def make_latex(self, inputs, bracketed=False):
        s = _latex(inputs[0])
        if isinstance(inputs[0], Function) and not inputs[0].definition > self:
            s = self._add_tex_brackets(s)
        return '-' + s

class MultiplicationDefinition(BinaryOperatorDefinition):
    def make_latex(self, inputs, bracketed=False):
        s = _latex(inputs[0]) + r'\cdot' + _latex(inputs[1])
        if bracketed:
            s = self._add_tex_brackets(s)
        return s

class DivisionDefinition(BinaryOperatorDefinition):
    def make_latex(self, inputs, bracketed=False):
        s = r'\frac{' + _latex(inputs[0]) + '}{' + _latex(inputs[1]) + '}'
        return s

class RootDefinition(FunctionDefinition):
    def make_latex(self, inputs, bracketed=False):
        if len(inputs) == 2:
            return r'\sqrt[' + _latex(inputs[1]) + ']{' + _latex(inputs[0]) + r'}'
        else:
            return r'\sqrt{' + _latex(inputs[0]) + r'}'

class LogbDefinition(FunctionDefinition):
    def make_latex(self, inputs, bracketed=False):
        b = _latex(inputs[1]) if len(inputs) == 2 else '10'
        return r'\log_{' + b + r'}{' + _latex(inputs[0]) + r'}'

class FactorialDefinition(FunctionDefinition):
    def make_latex(self, inputs, bracketed=False):
        s = _latex(inputs[0])
        if isinstance(inputs[0], _parse_time_funcs):
            s = self._add_tex_brackets(s)
        return r'{' + s + r'}!'

class ChooseDefinition(FunctionDefinition):
    def make_latex(self, inputs, bracketed=False):
        return _latex(inputs[0]) + r'\choose' + _latex(inputs[1])

class IntegralDefinition(FunctionDefinition):
    def make_latex(self, inputs, bracketed=False):
        dx = inputs[0].definition.args[0]
        dx = _latex_symbols.get(dx, dx)
        func = _latex(inputs[0], braces=False)
        if isinstance(inputs[0], CustomFunction):
            func = func.split('=', 1)[1]
        return r'\int_{' + _latex(inputs[1]) + r'}^{' + _latex(inputs[2]) + r'}' + func + r'\,d' + dx

class DerivativeDefinition(FunctionDefinition):
    def make_latex(self, inputs, bracketed=False):
        n = inputs[2] if len(inputs) == 3 else 1
        dx = inputs[0].definition.args[0]
        dx = _latex_symbols.get(dx, dx)
        if n == 1:
            numer = r'd'
            denom = r'd' + dx
        else:
            n = _latex(n)
            numer = r'd^{' + n + r'}'
            denom = r'd' + dx + r'^{' + n + r'}'
        func = _latex(inputs[0], braces=False)
        if isinstance(inputs[0], CustomFunction):
            func = func.split('=', 1)[1]
        return r'\frac{' + numer + r'}{' + denom \
               + r'}\Bigr|_{' + dx + r'=' + _latex(inputs[1]) \
               + r'}{\left[' + func + r'\right]}'


golden = 1.618033988749895 # golden ratio (1+√5)/2
_sqrt5 = math.sqrt(5)

def hypot(x, y):
    return math.sqrt(x*x + y*y)

def binomial(n, x, p):
    return math.comb(n, x) * pow(p, x) * pow(1-p, n-x)

def fibonacci(n):
    return int((golden**n - (-golden)**-n) / _sqrt5)

def integrate(f, a, b):
    return quad(f, a, b)[0]

def differentiate(f, x, n=1):
    return derivative(f, x, dx=1e-4, n=n)

def cartesian_to_polar(x, y):
    return [hypot(x, y), math.atan2(y, x)]

def polar_to_cartesian(r, theta):
    return [r*math.cos(theta), r*math.sin(theta)]

def cartesian_to_cylindrical(x, y, z):
    return [hypot(x, y), math.atan2(y, x), z]

def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x*x + y*y + z*z)
    return [r, math.atan2(y, x), math.acos(z/r)]

def cylindrical_to_cartesian(rho, phi, z):
    return [rho*math.cos(phi), rho*math.sin(phi), z]

def cylindrical_to_spherical(rho, phi, z):
    return [hypot(rho, z), phi, math.atan2(rho, z)]

def spherical_to_cartesian(r, theta, phi):
    return [r*math.sin(phi)*math.cos(theta), r*math.sin(theta)*math.sin(phi), r*math.cos(phi)]

def spherical_to_cylindrical(r, theta, phi):
    return [r*math.sin(phi), theta, r*math.cos(phi)]


def create_default_context():
    ctx = Context()
    ctx.add(
        # Constants
        Constant('π',   math.pi),
        Constant('pi',  math.pi, 'π'),
        Constant('e',   math.e),
        Constant('ϕ',   golden),
        Constant('phi', golden, 'ϕ'),
        Constant('∞',   math.inf),
        Constant('inf', math.inf, '∞'),

        # Basic Binary Operators
        BinaryOperatorDefinition('+', operator.add,     1, 0),
        BinaryOperatorDefinition('-', operator.sub,     1, 0),
        MultiplicationDefinition('*', operator.mul,     3, 0),
        DivisionDefinition      ('/', operator.truediv, 3, 0),
        BinaryOperatorDefinition('%', operator.mod,     3, 0),
        BinaryOperatorDefinition('^', operator.pow,     5, 1),
        BinaryOperatorDefinition(',', None, 0, 0),

        # Basic Functions
        NegationDefinition('neg',   'x', operator.neg),
        FunctionDefinition('abs',   'x', abs),
        FunctionDefinition('rad',   'θ', math.radians),
        FunctionDefinition('deg',   'θ', math.degrees),
        FunctionDefinition('round', 'x', round),
        FunctionDefinition('floor', 'x', math.floor),
        FunctionDefinition('ceil',  'x', math.ceil),
        FunctionDefinition('ans',   '',  lambda: ctx.ans),

        # Roots & Complex Functions
        RootDefinition    ('sqrt',  'x',  math.sqrt),
        RootDefinition    ('root',  'xn', lambda x, n: x**(1/n)),
        FunctionDefinition('hypot', 'xy', lambda x, y: hypot),

        # Trigonometric Functions
        FunctionDefinition('sin',  'θ', math.sin),
        FunctionDefinition('cos',  'θ', math.cos),
        FunctionDefinition('tan',  'θ', math.tan),
        FunctionDefinition('sec',  'θ', lambda x: 1/math.cos(x)),
        FunctionDefinition('csc',  'θ', lambda x: 1/math.sin(x)),
        FunctionDefinition('cot',  'θ', lambda x: 1/math.tan(x)),
        FunctionDefinition('asin', 'x', math.asin),
        FunctionDefinition('acos', 'x', math.acos),
        FunctionDefinition('atan', 'x', math.atan),

        # Hyperbolic Functions
        FunctionDefinition('sinh', 'x', math.sinh),
        FunctionDefinition('cosh', 'x', math.cosh),
        FunctionDefinition('tanh', 'x', math.tanh),

        # Exponential & Logarithmic Functions
        FunctionDefinition('exp',  'x',  math.exp),
        FunctionDefinition('ln',   'x',  math.log),
        LogbDefinition    ('log',  'x',  math.log10),
        LogbDefinition    ('logb', 'xb', math.log),

        # Combinatorial & Random Functions
        FactorialDefinition('fact',   'n',   math.factorial),
        FunctionDefinition ('perm',   'nk',  math.perm),
        ChooseDefinition   ('choose', 'nk',  math.comb),
        FunctionDefinition ('binom',  'nxp', binomial),
        FunctionDefinition ('fib',    'n',   fibonacci),
        FunctionDefinition ('rand',   '',    random),
        FunctionDefinition ('randr',  'ab',  lambda a, b: random() * (b - a) + a),

        # Calculus & etc.
        IntegralDefinition  ('int',    [f_arg('f', 1), 'a', 'b'], integrate),
        DerivativeDefinition('deriv',  [f_arg('f', 1), 'x'],      differentiate),
        DerivativeDefinition('nderiv', [f_arg('f', 1), 'x', 'n'], differentiate),

        # Coordinate System Conversion Functions
        FunctionDefinition('polar',  'xy',  cartesian_to_polar),
        FunctionDefinition('rect',   'rθ',  polar_to_cartesian),
        FunctionDefinition('crtcyl', 'xyz', cartesian_to_cylindrical),
        FunctionDefinition('crtsph', 'xyz', cartesian_to_spherical),
        FunctionDefinition('cylcrt', 'ρϕz', cylindrical_to_cartesian),
        FunctionDefinition('cylsph', 'ρϕz', cylindrical_to_spherical),
        FunctionDefinition('sphcrt', 'rθϕ', spherical_to_cartesian),
        FunctionDefinition('sphcyl', 'rθϕ', spherical_to_cylindrical),

        root=True
    )
    return ctx
