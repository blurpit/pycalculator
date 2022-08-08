from .context import Context
from .definitions import BinaryOperatorDefinition, FunctionDefinition, Function, FunctionArgument, \
    CustomFunction, Vector
from .tokenizer import tokenize


class _FuncInputsCtxState:
    def __init__(self, definition=None):
        self.definition = definition
        self.arg_index = 0

    def is_func_arg(self):
        return self.definition.is_func_arg(self.arg_index)


def parse(ctx:Context, tokens):
    stack = []
    output = []

    def peek():
        return stack[-1]

    def is_inside_func_arg():
        s = ctx.get_state()
        return s and s.is_func_arg()

    def increment_arg_index():
        s = ctx.get_state()
        if s: s.arg_index += 1

    def coalesce(definition):
        # Binary operator
        if isinstance(definition, BinaryOperatorDefinition):
            if len(output) < 2:
                raise SyntaxError("Not enough operands for binary operator '{}'.".format(definition.name))
            right = parse_operand(ctx, output.pop())
            left = parse_operand(ctx, output.pop())
            if definition.name == ',':
                # special case for , operator
                output.append(Vector.combine(left, right))
            else:
                output.append(Function(ctx, definition, [left, right]))

        # Function argument
        elif not new_func_definition and is_inside_func_arg():
            output.append(FunctionArgument(definition))

        # Normal function
        elif len(definition.args) > 0:
            if len(output) < 1:
                raise SyntaxError("{} expected {} argument(s), but {} were given."
                                  .format(definition.signature, len(definition.args), 0))
            inputs = parse_operand(ctx, output.pop())
            output.append(Function(ctx, definition, inputs))

        # Zero-input function
        else:
            output.append(Function(ctx, definition, []))

    # Check whether this expression is defining a new function
    new_func_definition = tokens[0] if len(tokens) >= 2 and tokens[1] == '=' else None
    if new_func_definition:
        del tokens[:2]
        ctx.push_context() # Push args to context so they are defined
        for arg in new_func_definition.args:
            ctx.set(arg, 0)

    # Check tokens are not empty
    if not tokens:
        raise SyntaxError("Expression is empty.")

    for token in tokens:
        # Binary operator or function
        if isinstance(token, FunctionDefinition):
            while stack and not token > peek():
                coalesce(stack.pop())

            if len(token.args) > 0:
                stack.append(token)
            else:
                # Do not push 0-input functions to the stack
                coalesce(token)

            if token.name == ',':
                increment_arg_index()

        # Operand or expression in parentheses
        else:
            top = peek() if stack else None
            inside_func = isinstance(top, FunctionDefinition) \
                          and not isinstance(top, BinaryOperatorDefinition)

            if inside_func:
                ctx.push_state(_FuncInputsCtxState(top))

            operand = parse_operand(ctx, token)
            output.append(operand)

            if isinstance(operand, Function) \
                    and not inside_func \
                    and isinstance(token, str) \
                    and (not top or top.name != ','):
                operand.bracketed = True

            if isinstance(operand, Vector):
                operand.bracketed = True

            if inside_func:
                ctx.pop_state()
                if isinstance(token, str):
                    # Close parenthesis. Coalesce one operator.
                    coalesce(stack.pop())
                else:
                    # Function without parenthesis. Coalesce by precedence.
                    while stack and not top > peek():
                        coalesce(stack.pop())

    while stack:
        coalesce(stack.pop())

    if len(output) > 1:
        raise SyntaxError("Leftover tokens {} after evaluation. Make sure all operators "
                          "and functions have valid inputs.".format(output))
    elif len(output) < 1:
        raise SyntaxError("Output was empty. Make sure all operators and functions have "
                          "valid inputs.")

    output = output[0]
    if new_func_definition:
        ctx.pop_context()
        new_func_definition.func = output
        output = CustomFunction(ctx, new_func_definition)

    return output

def parse_operand(ctx, token):
    if isinstance(token, str):
        return parse(ctx, tokenize(ctx, token))
    else:
        return token
