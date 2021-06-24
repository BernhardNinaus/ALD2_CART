from DecisionTreeClassifier import *
import pandas as pd
import json

titanic = pd.read_csv("data/titanic.csv")

target = titanic["Survived"]
features = titanic.loc[:, ["Pclass", "Sex", "Age"]]

# Pclass wird von numerischen zu muliclass geändert um gini richtig zu berechnen
features.Pclass = features.Pclass.map({1: "1st", 2: "2nd", 3: "3rd"})

tree_builder = DecisionTreeClassifier(target, features, max_depht=3)
tree_builder.buildDT()

# Tree als Text-Tree ausgeben
f = open("output.txt", "w")
f.write(str(tree_builder))
f.close()
# Tree als JSON speichern.
json_file_name = "output.json"
f = open(json_file_name, "w")
f.write(repr(tree_builder.tree))
f.close()


# Von JSON parsenen.
json_tree = DecissionTreeNode.fromJson(open(json_file_name).read())

# Predict neue Datensätze, geparst von JSON.
rose = pd.Series({'Pclass' : "1st",
        'Sex' : "female",
        'Age' : 19})

jack = pd.Series({'Pclass' : "3rd",
        'Sex' : "male",
        'Age' : 21})

print(f"Rose: {json_tree.predictSample(rose)*100:.2f}%", )
print(f"Jack: {json_tree.predictSample(jack)*100:.2f}%", )
