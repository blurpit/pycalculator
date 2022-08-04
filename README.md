
# Calculator

An overly complicated calculator library I made out of boredom.

## Install
```
pip install git+https://github.com/blurpit/pycalculator.git
```

Required python packages: numpy, scipy, matplotlib, latex, pdf2image.
[LaTeX installation](https://www.latex-project.org/) is required for latex rendering. Make sure the latex executables are added to the PATH environment variable.

# Usage
```py
import calc
ctx = calc.create_default_context()
ctx.push_context()
```
The context contains definitions for all binary operators, functions, variables, and
constants.

## Expression evaluation
### Simple expressions
```py
calc.evaluate(ctx, '9 + 10') # simple addition
> 19

# order of operations and parentheses
calc.evaluate(ctx, '2^(3+1)/12+1.5') 
> 2.833333333333333

# constants
calc.evaluate(ctx, 'e^-3 + e^2') 
> 7.438843167298513

# ans example
calc.evaluate(ctx, 'ans*2') 
> 14.877686334597026

# multiple outputs example
calc.evaluate(ctx, '2^1, 2^2, 2^3, 2^4, 2^5, 2^6')
> [2, 4, 8, 16, 32, 64]

# implicit multiplication is not implemented so don't try it.
calc.evaluate(ctx, '4(3+2)(6)')
> SyntaxError: Leftover tokens [4, (3+2), 6] after evaluation. Make sure 
all operators and functions have valid inputs.
```

### Using functions
```py
calc.evaluate(ctx, 'sin(2*pi/3), sqrt(3)/2')
> [0.8660254037844387, 0.8660254037844386]

calc.evaluate(ctx, 'logb(3^9, 3)')
> 9.0
```

### Creating functions
```py
# Defining a function using calc.evaluate() returns a CustomFunction object
f = calc.evaluate('f(x) = x^2 + 3*x + 5')
f
> f(x) = x^2+3*x+5

# The returned function f can be called like a regular python function
f(4)
> 33

# the f object can be added to the context, then it can be used in 
# future expressions
ctx.add(f)
calc.evaluate(ctx, 'f(3+1)')
> 33

# multiple arguments example
f = calc.evaluate(ctx, 'f(x, y) = x^3 * y^2')
ctx.add(f)
f(3, 2)
> 108
```

### Creating constants
Constants are just treated as 0-argument functions, so they work similarly to defining a
function as above.
```py
foo = calc.evaluate(ctx, 'foo = 7*pi/3')
foo
> foo = 7*π/3
foo()
> 7.330382858376184

ctx.add(foo)
calc.evaluate(ctx, 'foo*3/7')
> 3.141592653589793

# since constants are 0-arg functions, they are calculated separately for each 
# time they are evaluated
foo = calc.evaluate(ctx, 'foo = round(rand()*100)')
for _ in range(10):
    print(foo(), end=' ')
> 48 6 70 69 82 84 9 56 52 80 

```

### Function arguments
Some functions (integrals, for example) take a function as an argument. New functions can
be defined inside the argument, or an existing function in the context can be referenced
without calling it.
```py
# defining the function inside an integral
calc.evaluate(ctx, 'int(f(t)=2*t, 0, 5)')
> 25.0

# referencing existing functions
calc.evaluate(ctx, 'int(sin, 0, 3*pi/4)')
> 1.7071067811865472

f = calc.evaluate(ctx, 'f(t) = (t+2)*(t-3)')
ctx.add(f)
calc.evaluate(ctx, 'int(f, -2, 2)')
> -18.666666666666668
```

## LaTeX
Expressions and custom functions can be converted to a LaTeX string using `calc.latex()`.
```py
calc.latex(ctx, '3/4*a')
> '{\frac{{3}}{{4}}}\cdot{a}'

calc.latex(ctx, 'int(f(t)=3*t/4+6, -5, 5)')
> '\int_{{-{5}}}^{{5}}{{\frac{{{3}\cdot{t}}}{{4}}}+{6}}\,dt'

calc.latex(ctx, 'f(x) = 3*x+6')
> '{f\left({x}\right)}={{{3}\cdot{x}}+{6}}'
```

LaTeX can also be converted into a png image using `calc.latex_to_image()`. The returned value will
be a PIL Image object.
```py
tex = calc.latex(ctx, 'int(f(t)=3*t/4+6, -5, 5)')
img = calc.latex_to_image(tex, color='black', transparent=False)
img.save('mytex.png')
```
![](https://i.imgur.com/Z0Cxy6Q.png)

## Graphing
1-dimensional functions (1 input and 1 output) can be graphed using `calc.graph()`. The returned value is a matplotlib Figure.
`savefig_bytesio()` and `savefig_png()` can be used to save the figure.
```py
fig = calc.graph(ctx, 'f(x)=1/25*(x-8)*(x-2)*(x+8)')
fig.show()
```
![](https://i.imgur.com/J0CuEkr.png)

`calc.graph()` accepts 4 types of function definitions:

1. explicit signature `foo(z) = 2*z^3`
2. y equals `y = 2*x^3`
3. nothing `2*x^3`
4. a function object from `calc.evaluate()` or `ctx.get()`

For 2 and 3, the variable used in the function must be `x`.

## Saving to JSON
Contexts (minus the root context) can be saved to .json files using `calc.save_contexts()` and loaded using `calc.load_contexts()`.

```py
ctx = calc.create_default_context()
ctx.push_context()

ctx.add(calc.evaluate(ctx, 'f(x) = 3*x^2 + 4'))
ctx.add(calc.evaluate(ctx, 'g(x) = 3/2*x^3 + 4*x - 1'))
ctx.push_context()
ctx.set('foo', 75.24623)

calc.save_contexts(ctx, 'saved_math.json')
```
```json
[{"f": "f(x) = 3*x^2+4", "g": "g(x) = 3/2*x^3+4*x-1"}, {"foo": "75.24623"}]
```

```py
ctx = calc.create_default_context()
# note that a new context is not pushed before loading

calc.load_contexts(ctx, 'saved_math.json')
calc.evaluate(ctx, 'f(foo), g(foo)')
> [16989.9853876387, 639365.6665474502]
```

## Contexts
A `Context` object has a stack of contexts that can be pushed or popped to create nested
scopes and redefine things without deleting them. Example:
```py
ctx = calc.create_default_context()

ctx.push_context()
ctx.set('x', 12)
print(ctx.get('x'))

ctx.push_context()
ctx.set('x', 42)
print(ctx.get('x'))

ctx.pop_context()
print(ctx.get('x'))

ctx.pop_context()
print(ctx.get('x'))
```
```
12
42
12
ContextError: 'x' is undefined.
```

The first context in the stack is the root context, and cannot be modified without setting `root=True` in `ctx.add()` or `ctx.set()`.

### The default context
`calc.create_default_context()` defines the following:

| Definition | Type | Description | 
| - | - | - |
| `+` | Binary operator | Add |
| `-` | Binary operator | Subtract |
| `*` | Binary operator | Multiply |
| `/` | Binary operator | Divide |
| `%` | Binary operator | Remainder |
| `^` | Binary operator | Exponent |
| `,` | Binary operator | Argument/output separator |
| `π` or `pi` | Constant | 3.14159... |
| `e` | Constant | 2.71828... |
| `ϕ` or `phi` | Constant | 1.61803... |
| `∞` or `inf` | Constant | Infinity |
| `ans` | Constant | Result of previous evaluation |
| `-x` | Unary operator | Negation |
| `abs(x)` | Function | Absolute value |
| `rad(θ)` | Function | Degrees -> radians |
| `deg(θ)` | Function | Radians -> degrees |
| `round(x)` | Function | Round to nearest int |
| `floor(x)` | Function | Round down to nearest int |
| `ceil(x)` | Function | Round up to nearest int |
| `sqrt(x)` | Function | Square root |
| `root(x, n)` | Function | n-th root |
| `hypot(x, y)` | Function | `sqrt(x^2 + y^2)` |
| `sin(θ)` | Function | Sine |
| `cos(θ)` | Function | Cosine |
| `tan(θ)` | Function | Tangent |
| `sec(θ)` | Function | Secant |
| `csc(θ)` | Function | Cosecant |
| `cot(θ)` | Function | Cotangent |
| `asin(x)` | Function | Inverse sine |
| `acos(x)` | Function | Inverse cosine |
| `atan(x)` | Function | Inverse tangent |
| `sinh(x)` | Function | Hyperbolic sine |
| `cosh(x)` | Function | Hyperbolic cosine |
| `tanh(x)` | Function | Hyperbolic tangent |
| `exp(x)` | Function | `e^x` |
| `ln(x)` | Function | Natural logarithm |
| `log(x)` | Function | Base 10 logarithm |
| `logb(x, b)` | Function | Base b logarithm |
| `fact(n)` | Function | Factorial |
| `perm(n, k)` | Function | Permutations |
| `choose(n, k)` | Function | Combinations |
| `binom(n, x, p)` | Function | Binomial probability |
| `fib(n)` | Function | n-th Fibonacci number |
| `rand()` | Function | Random between 0 and 1 |
| `randr(a, b)` | Function | Random between a and b |
| `int(f(_), a, b)` | Function | Definite integral
| `deriv(f(_), x)` | Function | Derivative evaluated at x |
| `nderiv(f(_), x)` | Function | n-th Derivative evaluated at x |

# Other stuff

## Todo list
i.e. things i'm not gonna bother with
1. Implicit multiplication
2. Add more than literally zero comments lol
3. Fix my mess of a life
4. Nested lists & passing a list as one argument
5. Functions with arbitrary number of arguments
6. Returning functions then calling them
7. Lists as function arguments
8. Delete this whole thing and just use a WolframAlpha API like a smart person