import numpy as np

class DecisionTreeClassifier:
    def __init__(self):
        self._root = None

    class Node:
        __slots__ = '_element', '_left', '_right'

        def __init__(self, element, left=None, right=None):
            self._element = element
            self._left = left
            self._right = right

    def gini(self, leaf1_zero, leaf1_one, leaf2_zero, leaf2_one):
        total_leaf1 = leaf1_zero + leaf1_one
        total_leaf2 = leaf2_zero + leaf2_one

        gini_leaf1 = 1 - (leaf1_zero/total_leaf1)^2 - (leaf1_one/total_leaf1)^2
        gini_leaf2 = 1 - (leaf2_zero/total_leaf2)^2 - (leaf2_one/total_leaf2)^2

        return (total_leaf1/(total_leaf1 + total_leaf2)) * gini_leaf1 + (total_leaf2/(total_leaf1 + total_leaf2)) * gini_leaf2

    def make_tree(self, X, y):
        ginis = []
        for col in range(X.shape[1]):
            print(X[:, col])



if __name__ == '__main__':
    dtc = DecisionTreeClassifier()
    arr = np.array([[1, 2], [2, 3], [3, 4]])
    dtc.make_tree(arr, 1)


