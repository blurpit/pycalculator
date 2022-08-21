from contextlib import contextmanager
from contextlib import contextmanager
from typing import Any

from .definitions import BinaryOperatorDefinition, FunctionDefinition, CustomFunction


class ContextError(Exception):
    pass

class Context:
    def __init__(self):
        self.parent = None
        self.ctx_stack = [dict()]
        self.state_stack = []
        self.ans = 0

    def add(self, *items, root=False):
        for item in items:
            if isinstance(item, CustomFunction):
                item = FunctionDefinition(item.definition.name, item.definition.args, item)
            self.set(item.name, item, root)

    def get(self, name, default:Any=ContextError):
        for i in range(len(self.ctx_stack)-1, -1, -1):
            result = self.ctx_stack[i].get(name, ContextError)
            if result is not ContextError:
                return result

        if default is ContextError:
            raise ContextError("'{}' is undefined.".format(name))
        else:
            return default

    def __contains__(self, name):
        for i in range(len(self.ctx_stack)-1, -1, -1):
            if name in self.ctx_stack[i]:
                return True
        return False

    def set(self, name, val, root=False):
        if not root:
            if len(self.ctx_stack) < 2:
                raise ContextError("Cannot add to context.")
            elif name in self.ctx_stack[0]:
                raise ContextError("Cannot override '{}'.".format(name))
        self.ctx_stack[-1][name] = val

    def keys(self):
        result = set()
        for i in range(len(self.ctx_stack)):
            result.update(self.ctx_stack[i])
        return result

    def push_context(self):
        self.ctx_stack.append(dict())

    def pop_context(self):
        if len(self.ctx_stack) > 1:
            del self.ctx_stack[-1]

    @contextmanager
    def with_context(self):
        try:
            self.push_context()
            yield
        finally:
            self.pop_context()

    def clear_contexts(self):
        del self.ctx_stack[1:]

    def get_state(self):
        return self.state_stack[-1] if self.state_stack else None

    def push_state(self, state):
        self.state_stack.append(state)

    def pop_state(self):
        if self.state_stack:
            self.state_stack.pop()

    def clear_states(self):
        self.state_stack.clear()

    def is_binary_operator(self, op):
        if isinstance(op, str):
            return isinstance(self.get(op, None), BinaryOperatorDefinition)
        else:
            return isinstance(op, BinaryOperatorDefinition)

    def is_function(self, func):
        if isinstance(func, str):
            return isinstance(self.get(func, None), FunctionDefinition)
        else:
            return isinstance(func, FunctionDefinition)

    def __bool__(self):
        # treat like object, not dictionary
        return True

    def __str__(self):
        s = 'Context('
        for i, ctx in enumerate(self.ctx_stack):
            s += '\n\t' + str(i) + ': ' + ' '.join(ctx.keys())
        return s + '\n)'
