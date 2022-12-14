from .context import Context
from .definitions import Variable, FunctionDefinition, Constant, BinaryOperatorDefinition


def tokenize(ctx:Context, exp):
    tokens = []
    pos = 0

    # Treat a comma separated expression separately
    comma_split = split_commas(ctx, exp)
    if len(comma_split) > 1:
        comma = ctx.get(',')
        for token in comma_split:
            if token == ',':
                tokens.append(comma)
            else:
                tokens.append(token)
        return tokens

    with ctx.with_context():
        # Split function signature from the rest of the expression
        custom_definition, pos = tokenize_signature(exp)
        if custom_definition:
            tokens.append(custom_definition)
            tokens.append('=')
            # Define the args in the context. Value is unnecessary.
            for arg in custom_definition.args:
                ctx.set(arg, 0)

        while pos < len(exp):
            token, pos = next_token(ctx, exp, pos)
            if token != [] and token != ['']:
                tokens.extend(token)

    add_implicit_mul(ctx, tokens)
    return tokens

def next_token(ctx:Context, text, pos):
    ch = text[pos]
    if ch is None:
        return None
    elif ch == '(':
        return tokenize_parenthesis(text, pos)
    elif ch.isdigit() or ch == '.':
        return tokenize_number(text, pos)
    elif ch == '-':
        # special case for unary and binary minus sign
        return tokenize_negation(ctx, text, pos)
    elif ctx.is_binary_operator(ch):
        return tokenize_binary_operator(ctx, text, pos)
    elif is_identifier(ch):
        return tokenize_alpha(ctx, text, pos)
    elif ch == ')':
        raise SyntaxError("Mismatching parenthesis (close > open)")
    else:
        msg = f"Unknown character '{ch}'."
        if ch == '|': msg += "\n(For absolute value, use 'abs(x)' instead of '|x|')"
        elif ch == '!': msg += "\n(For factorials, use 'fact(n)' instead of 'n!')"
        elif ch == '=': msg += "\n(Functions may only be defined at the start of an expression or inside a function argument)"
        raise SyntaxError(msg)

def tokenize_parenthesis(text, pos):
    start = pos + 1
    parens = 0
    while text[pos] != ')' or parens > 1:
        if text[pos] == '(': parens += 1
        elif text[pos] == ')': parens -= 1
        pos += 1
        if pos >= len(text):
            raise SyntaxError("Mismatching parenthesis (open > close).")
    return [text[start:pos]], pos + 1

def tokenize_number(text, pos):
    start = pos
    while pos < len(text) and (text[pos] == '.' or text[pos].isnumeric()):
        pos += 1
    num = float(text[start:pos])
    if num % 1 == 0:
        num = int(num)
    return [num], pos

def tokenize_alpha(ctx, text, pos):
    start = pos
    while pos < len(text) and is_identifier(text[pos]):
        pos += 1
    tokens = split_alpha(ctx, text[start:pos])
    for i, name in enumerate(tokens):
        tokens[i] = ctx.get(name, None) or Variable(ctx, name)
    return tokens, pos

def split_alpha(ctx, text):
    max_len = 1
    for i in range(1, len(text) + 1):
        if text[:i] in ctx:
            max_len = i
    result = [text[:max_len]]
    if max_len < len(text):
        result += split_alpha(ctx, text[max_len:])
    return result

def tokenize_binary_operator(ctx, text, pos):
    return [ctx.get(text[pos])], pos + 1

def tokenize_negation(ctx, text, pos):
    prev = text[pos - 1] if pos > 0 else None
    if prev is None or prev == '=' or ctx.is_binary_operator(prev):
        return [ctx.get('neg')], pos + 1
    else:
        return tokenize_binary_operator(ctx, text, pos)


def tokenize_signature(text):
    parts = text.split('=', 1)
    if len(parts) == 2:
        definition = parse_signature(parts[0])
        if definition:
            return definition, len(parts[0]) + 1
    return None, 0

def parse_signature(text):
    parts = text.split('(')
    if len(parts) > 2: # invalid
        raise SyntaxError("'{}' is not a valid function signature.".format(text))
    elif len(parts) == 1: # constant
        name, args = text, []
        use_cache = True
    else: # function
        name, args = parts
        args = args.lstrip('(').rstrip(')').split(',')
        if args == ['']: # 0-arg function
            args = []
        use_cache = False

    if not is_identifier(name):
        raise ValueError("'{}' is not a valid function name.".format(name))
    for arg in args:
        if not is_identifier(arg):
            raise ValueError("'{}' is not a valid argument name.".format(arg))

    return FunctionDefinition(
        name, args, None,
        use_cache=use_cache
    )

def split_commas(ctx, text):
    """ Split a string by commas that are not inside parenthesis """
    comma = ctx.get(',')
    parens = 0
    start = 0
    result = []
    for i, c in enumerate(text):
        if c == '(':
            parens += 1
        elif c == ')':
            parens -= 1
        elif c == ',' and parens == 0:
            result.append(text[start:i])
            result.append(comma)
            start = i + 1
    result.append(text[start:])
    return result

def is_identifier(name):
    return name.isalpha()

def add_implicit_mul(ctx, tokens):
    mul = ctx.get('*')
    i = 1 if len(tokens) < 3 or tokens[1] != '=' else 3 # skip func definition

    while i < len(tokens):
        curr = tokens[i]
        prev = tokens[i - 1]
        if is_implicit_mul(prev, curr):
            tokens.insert(i, mul)
            i += 1
        i += 1

def is_implicit_mul(prev, curr):
    if isinstance(prev, BinaryOperatorDefinition) or isinstance(curr, BinaryOperatorDefinition):
        return False
    if isinstance(prev, FunctionDefinition):
        return len(prev.args) == 0
    return True
