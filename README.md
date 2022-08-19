
# Calculator

An overly complicated calculator library I made out of boredom.

## Install
```
pip install git+https://github.com/blurpit/pycalculator.git
```

Required python packages: numpy, scipy, matplotlib, latex, pdf2image.
A [LaTeX installation](https://www.latex-project.org/) is required for latex rendering. Make sure the latex executables are added to the PATH environment variable.

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
# simple addition
calc.evaluate(ctx, '9 + 10')
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
> (2, 4, 8, 16, 32, 64)

# implicit multiplication example
calc.evaluate(ctx, '4(3+2)(6)')
> 120
```

### Using functions
```py
calc.evaluate(ctx, 'sin(2pi/3), sqrt(3)/2')
> (0.8660254037844387, 0.8660254037844386)

calc.evaluate(ctx, 'logb(3^9, 3)')
> 9.0

# Parentheses can be omitted, in which case functions 
# have precedence between multiplication and addition.
calc.evaluate(ctx, '1/2sin2/3pi^2+10')
> 10.146111730949718
```

### Creating functions
```py
# Defining a function using calc.evaluate() returns a CustomFunction object
f = calc.evaluate('f(x) = x^2 + 3x + 5')
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
Constants are treated as 0-argument functions, so they work similarly to defining a
function as above.
```py
foo = calc.evaluate(ctx, 'foo = 7pi/3')
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
ctx.add(foo)
for _ in range(10):
    print(calc.evaluate(ctx, 'foo'), end=' ')
> 66 3 23 66 81 34 20 11 84 41 

```

### Function arguments
Some functions (integrals, for example) take a function as an argument. New functions can
be defined inside the argument, or an existing function in the context can be referenced
without calling it.
```py
# defining the function inside an integral
calc.evaluate(ctx, 'int(f(t)=2t, 0, 5)')
> 25.0

# referencing existing functions
calc.evaluate(ctx, 'int(sin, 0, 3pi/4)')
> 1.7071067811865472

f = calc.evaluate(ctx, 'f(t) = (t+2)(t-3)')
ctx.add(f)
calc.evaluate(ctx, 'int(f, -2, 2)')
> -18.666666666666668
```

### Vectors & Matrices
The `v(*x)` and `mat(*v)` functions are used to create vectors and matrices respectively.
```py
calc.evaluate(ctx, '2 * v(1, 1+1, 4-1)')
> (2, 4, 6)

# the 'v' when creating vectors can sometimes be omitted, such as inside mat()
calc.evaluate(ctx, 'mat((2^3, 2^4), (2^5, 2^6))')
> mat((8, 16), (32, 64))
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
tex = calc.latex(ctx, 'int(f(t)=3t/4+6, -5, 5)')
img = calc.latex_to_image(tex, color='black', transparent=False)
img.save('mytex.png')
```
![](https://i.imgur.com/Z0Cxy6Q.png)

## Graphing
1-dimensional functions (1 input and 1 output) can be graphed using `calc.graph()`. The returned value is a matplotlib Figure.
`savefig_bytesio()` and `savefig_png()` can be used to save the figure.
```py
fig = calc.graph(ctx, 'f(x)=1/25(x-8)(x-2)(x+8)')
fig.show()
```
![](https://i.imgur.com/J0CuEkr.png)

`calc.graph()` accepts 4 types of function definitions:

1. explicit signature `foo(z) = 2z^3`
2. y equals `y = 2x^3`
3. nothing `2x^3`
4. a function object from `calc.evaluate()` or `ctx.get()`

For 2 and 3, the variable used in the function must be `x`.

## Contexts
A context is a set of defined symbols for the evaluator. These include operators, functions,
constants, variables, etc.
A `calc.Context` object has a stack of contexts that can be pushed or popped to create nested
scopes and redefine things without deleting them. Use `ctx.set(name, item)` to add to the context, 
or `ctx.add(item)` if the item is a `FunctionDefinition`, `Constant`, or `CustomFunction`.

Example:
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

### Function Definitions
A `FunctionDefinition` defines the name and arguments for a function.
```py
from calc.definitions import FunctionDefinition
ctx = calc.create_default_context()
ctx.push_context()

def my_function(a, b, c):
    return (a + b) ** c

ctx.add(
    # 'abc' can be used as a shorthand for ['a', 'b', 'c'].
    FunctionDefinition('myfun', 'abc', my_function)
)
calc.evaluate(ctx, 'myfun(5, -2, 7)')
> 2187

# BinaryOperatorDefinitions work similarly.
from calc.definitions import BinaryOperatorDefinition

def bin_and(a, b):
    return a & b

ctx.add(
    # The 4 and 0 are precedence and associativity respectively.
    # Lower number = higher precedence.
    # 0 for left-to-right associativity, 1 for right-to-left associativity.
    BinaryOperatorDefinition('&', bin_and, 4, 0)
)
calc.evaluate(ctx, '742 & 357')
> 100

# If you want a function argument, wrap the argument in f_arg().
from calc.definitions import FunctionDefinition, f_arg

def my_function(f, a, b):
    return sum(f(i) for i in range(a, b))

ctx.add(
    # f_arg() takes the name of the argument, and the number of arguments 
    # the expected function takes
    FunctionDefinition('myfun', [f_arg('f', 1), 'a', 'b'], my_function)
)
calc.evaluate(ctx, 'myfun(f(x)=2x, 0, 100)')
> 9900
```

### Saving to JSON
Contexts (minus the root context) can be saved to .json files using `calc.save_contexts()` and loaded using `calc.load_contexts()`.

```py
ctx = calc.create_default_context()
ctx.push_context()

ctx.add(calc.evaluate(ctx, 'f(x) = 3x^2 + 4'))
ctx.add(calc.evaluate(ctx, 'g(x) = 3/2x^3 + 4x - 1'))
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
> (16989.9853876387, 639365.6665474502)
```

### The default context
`calc.create_default_context()` defines all of the following:

#### Binary Operators
| Definition        | Description                    | 
|-------------------|--------------------------------|
| `+`               | Add                            |
| `-`               | Subtract                       |
| `*`               | Multiply                       |
| `/`               | Divide                         |
| `%`               | Remainder                      |
| `^`               | Exponent                       |
| `,`               | Argument/output separator      |

#### Constants
| Definition        | Description                    |
|-------------------|--------------------------------|
| `π` or `pi`       | 3.14159...                     |
| `e`               | 2.71828...                     |
| `ϕ` or `phi`      | 1.61803...                     |
| `∞` or `inf`      | Infinity                       |
| `ans`             | Result of previous evaluation  |

#### Basic Functions
| Definition        | Description                    |
|-------------------|--------------------------------|
| `-x`              | Negation                       |
| `abs(x)`          | Absolute value                 |
| `rad(θ)`          | Degrees -> radians             |
| `deg(θ)`          | Radians -> degrees             |
| `round(x)`        | Round to nearest int           |
| `floor(x)`        | Round down to nearest int      |
| `ceil(x)`         | Round up to nearest int        |

#### Roots & Complex Functions
| Definition        | Description                    |
|-------------------|--------------------------------|
| `sqrt(x)`         | Square root                    |
| `root(x, n)`      | n-th root                      |
| `hypot(x, y)`     | `sqrt(x^2 + y^2)`              |

#### Trigonometric Functions
| Definition        | Description                    |
|-------------------|--------------------------------|
| `sin(θ)`          | Sine                           |
| `cos(θ)`          | Cosine                         |
| `tan(θ)`          | Tangent                        |
| `sec(θ)`          | Secant                         |
| `csc(θ)`          | Cosecant                       |
| `cot(θ)`          | Cotangent                      |
| `asin(x)`         | Inverse sine                   |
| `acos(x)`         | Inverse cosine                 |
| `atan(x)`         | Inverse tangent                |
| `sinh(x)`         | Hyperbolic sine                |
| `cosh(x)`         | Hyperbolic cosine              |
| `tanh(x)`         | Hyperbolic tangent             |

#### Exponential & Logarithmic Functions
| Definition        | Description                    |
|-------------------|--------------------------------|
| `exp(x)`          | `e^x`                          |
| `ln(x)`           | Natural logarithm              |
| `log(x)`          | Base 10 logarithm              |
| `logb(x, b)`      | Base b logarithm               |

#### Combinatorial & Random Functions
| Definition        | Description                    |
|-------------------|--------------------------------|
| `fact(n)`         | Factorial                      |
| `perm(n, k)`      | Permutations                   |
| `choose(n, k)`    | Combinations                   |
| `binom(n, x, p)`  | Binomial probability           |
| `fib(n)`          | n-th Fibonacci number          |
| `rand()`          | Random between 0 and 1         |
| `randr(a, b)`     | Random between a and b         |

#### Calculus
| Definition           | Description                    |
|----------------------|--------------------------------|
| `int(f(_), a, b)`    | Definite integral              |
| `deriv(f(_), x)`     | Derivative evaluated at x      |
| `nderiv(f(_), x, n)` | n-th Derivative evaluated at x |

#### Vectors & Matrices
| Definition      | Description                    |
|-----------------|--------------------------------|
| `v(*x)`         | Create a vector                |
| `mat(*v)`       | Create a matrix                |
| `I(n)`          | NxN identity matrix            |
| `dot(v, w)`     | Vector dot product             |
| `mag(v)`        | Vector length                  |
| `norm(v)`       | Normalized vector              |
| `shape(M)`      | Matrix shape (rows, cols)      |
| `mrow(M, r)`    | Get matrix row vector          |
| `mcol(M, c)`    | Get matrix column vector       |
| `mpos(M, r, c)` | Get matrix value at (row, col) |
| `vi(v, i)`      | Get vector value at index i    |

#### Linear Algebra
| Definition  | Description             |
|-------------|-------------------------|
| `det(M)`    | Matrix determinant      |
| `rank(M)`   | Matrix rank             |
| `inv(M)`    | Matrix inverse          |
| `kernel(M)` | Matrix null space       |
| `lu(M)`     | Matrix LU decomposition |
| `svd(M)`    | Matrix SVD              |


# Other stuff

## Todo list
* Returning functions then calling them
* Delete this whole thing and just use a WolframAlpha API like a smart person
* ~~Nested lists & passing a list as one argument~~
* ~~Functions with arbitrary number of arguments~~
* ~~Lists as function arguments~~
* ~~Implicit multiplication~~
