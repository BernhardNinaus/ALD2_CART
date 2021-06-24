from DecissionTreeNode import *
import numpy as np
import pandas as pd
from typing import *

class DecisionTreeClassifier:
    def __init__(self, target: pd.DataFrame, features: pd.DataFrame, max_depht: int = 3):
        self._target = target
        self._features = features

        self.tree = DecissionTreeNode()

        self.max_depht = max_depht

    def buildDT(self):
        '''Builds the decision tree recursively'''
        self._build_node(self.tree, self._target, self._features)


    def _build_node(self,
            node: DecissionTreeNode,
            target_values: pd.DataFrame, 
            feature_values: pd.DataFrame, 
            depht: int = 0):
        '''Es wird der übergebene Node mit den übergegebenen Daten berechnet.'''
        # Hole bestes Feature und Combination für diese Daten.
        gini_impurity, feature_name, combination = DecisionTreeClassifier._evaluate_node(target_values, feature_values)

        # Es konnte keine Decission erzeugt werden, Daten vielleicht schon 100% pure.
        if gini_impurity == None:
            return

        # Es wird die eigene Impurity berechnet.
        node_gini_value = DecisionTreeClassifier._gini_impurity(*target_values.value_counts().tolist())


        # Es wurde was gefunden, es soll die Decission nur eingebaut werden, wenn Daten reiner werden.
        if gini_impurity < node_gini_value and depht < self.max_depht:
            node.feature = feature_name
            node.combination = combination

            # Todo eq 1 entfernen
            # Berechne Wahrscheinlickeit für Ausgeben.
            node.chance = target_values[target_values.eq(1)].count() / target_values.count()

            # Mask für neue linken und rechten Bucket vorbereitet.
            data_left = feature_values[node.feature]
            if feature_values[node.feature].dtype != "O":
                data_left = data_left.le(node.combination)
            else:
                data_left = data_left.eq(node.combination)

            data_right = [not elem for elem in data_left]

            # Daten für linken und rechten Bucket vorbereiten.
            new_target_left = target_values[data_left]
            new_features_left = feature_values[data_left]
            new_target_right = target_values[data_right]
            new_features_right = feature_values[data_right]

            # Es werden neue Nodes erstllt und mit den neuen Daten weiterberechnet.
            node.left = DecissionTreeNode()
            self._build_node(node.left, new_target_left, new_features_left, depht + 1)

            node.right = DecissionTreeNode()
            self._build_node(node.right, new_target_right, new_features_right, depht + 1)

            # todo eq 1 entfernen
            # Wenn beim neuen Node kein Feature, und somit auch keine Decission, gefunden wurde,
            # dann nur noch Wahrscheinlichkeit berechnen.
            if node.left.feature == None:
                node.left.chance = target_values[target_values.eq(1) & data_left].count() \
                                        / target_values[data_left].count()

            if node.right.feature == None:
                node.right.chance = target_values[target_values.eq(1) & data_right].count() \
                                        / target_values[data_right].count()

    def __str__(self):
        return str(self.tree)

    @staticmethod
    def _evaluate_node(target_values: pd.DataFrame, feature_values: pd.DataFrame) \
        -> Tuple[float, str, Any]:
        '''
        Gibt beste gini_impurity, feature_name and combination als Tuple zurück.\r\n
        Wenn None, dann konnte keiner ermittelt werden.
        '''
        gini_impurity = None
        feature_name = None
        combination = None

        # Für jede Spalte wird die Gini Impurity berechnet und der beste Wert wird zurückgegeben.
        for current_feature_name in feature_values.columns:
            # Berechne beste gini impurity für akutelle Spalte.
            new_gini_impurity, new_combination = DecisionTreeClassifier._calculate_gini(
                                                    current_feature_name,
                                                    target_values,
                                                    feature_values)

            # Nur wenn neue Gini Impurity kleiner ist, dann soll diese verwendet werden.
            if not gini_impurity or (new_gini_impurity and new_gini_impurity < gini_impurity):
                gini_impurity = new_gini_impurity
                feature_name = current_feature_name
                combination = new_combination

        # Gefunden werte zurückgeben, None wenn nichts gefunden wurde.
        return gini_impurity, feature_name, combination

    @staticmethod
    def _calculate_gini(feature_name: str, target_values: pd.DataFrame, feature_values: pd.DataFrame) \
        -> Tuple[float, Any]:
        '''Gini Berechnung, nur für das eine Feature'''
        gini_impurity = None
        combination = None

        # Es werden zuerst die feautre Daten vorbereitet #
        # und festgestellt ob es ein numerisches Feature ist.
        feature_data = feature_values[feature_name]
        is_numeric = feature_data.dtype != "O"

        for item in np.sort(feature_data.unique()):
            # Vergleichsfunktion aussuchen.
            fnc = feature_data.le if is_numeric else feature_data.eq

            # Hier werden virtuelle linke und rechte Buckets erstellt,
            # um die Häufigkeit des Targets berechnen zu können.
            bucket_left = target_values[fnc(item)].value_counts().tolist()
            bucket_right = target_values[~fnc(item)].value_counts().tolist()

            if sum(bucket_left) == 0 or sum(bucket_right) == 0:
                # Wenn im linken oder rechten bucket nichts drinnen ist,
                # dann würde diese entscheidung im baum keinen unterschied machen (Gini => unendlich)
                # Es wäre alles im linken oder rechten Bucket.
                continue

            # Es werden Werte hinzugefügt, nur für Funktionsaufruf, 
            # wenn is in einem Bucket eine 100% Wahrscheinlickeit gibt.
            if len(bucket_left) == 1:
                bucket_left.append(0)
            if len(bucket_right) == 1:
                bucket_right.append(0)

            new_gini_impurity = DecisionTreeClassifier._gini_impurity_total(*(bucket_left + bucket_right))

            # Wenn der neue Gini-Wert einer anderen Combination besser ist, dann wird diese ausgewählt.
            if not gini_impurity or new_gini_impurity < gini_impurity:
                gini_impurity = new_gini_impurity
                combination = item

        return gini_impurity, combination

    @staticmethod
    def _gini_impurity_total(a=0, b=0, c=0, d=0):
        return ((a+b) * DecisionTreeClassifier._gini_impurity(a, b) + \
                (c+d) * DecisionTreeClassifier._gini_impurity(c, d)) \
            / (a + b + c + d)

    @staticmethod
    def _gini_impurity(a=0, b=0):
        return 1 - (np.square(a/(a+b)) + np.square(b/(a+b)))