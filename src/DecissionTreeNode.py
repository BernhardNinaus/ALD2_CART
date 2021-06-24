import pandas as pd
from typing import *

class DecissionTreeNode:
    __slots__ = ["left", "right", "chance", "feature", "combination"]
    
    def __init__(self):
        self.left: DecissionTreeNode = None
        self.right: DecissionTreeNode = None
        self.chance: float = None

        self.feature: str = None
        self.combination: Any = None

    def predictSample(self, data: pd.Series):
        if not self.feature:
            return self.chance

        test = (data[self.feature] <= self.combination) if self._is_numeric() else data[self.feature] == self.combination

        if test:
            return self.left.predictSample(data)
        else:
            return self.right.predictSample(data)
    
    def _is_numeric(self):
        return isinstance(self.combination, int) or isinstance(self.combination, float)

    def _display_aux(self):
        '''
        Returns list of strings, width, height, and horizontal coordinate of the root.\r\n
        https://stackoverflow.com/a/54074933/12452156
        '''
        if self.right is None and self.left is None:
            line = f"{round(self.chance * 100,1)}%"
            width = len(line)
            return [line], width, 1, width // 2

        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self._node_text()
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self._node_text()
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        left, n, p, x = self.left._display_aux() if self.left != None else self._left_text()
        right, m, q, y = self.right._display_aux() if self.right != None else self._right_text()
        s = '%s' % self._node_text()

        u = len(s)
        first_line = (x + 1) * ' '  + (n - x - 1) * '_'         + s  + y * '_'        + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' '             + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def _node_text(self):
        return f"_<- {self.feature} {self.combination}|surv:{round(self.chance * 100,1)}%"

    def __str__(self):
        lines, *_ = self._display_aux()
        return "\r".join(lines)