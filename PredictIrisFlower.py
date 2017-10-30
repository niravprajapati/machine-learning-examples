import numpy as np
from sklearn.datasets import load_iris
from sklearn import  tree

# load Iris flower data
iris = load_iris()
print iris.target_names
print iris.data
print iris.target
print iris.DESCR

test_idx = [0,50,100]

# training data
training_target = np.delete(iris.target, test_idx)
print "training_target"
print training_target

train_data = np.delete(iris.data,test_idx,axis=0)
print "train_data"
print train_data

# testing data
test_target = iris.target[test_idx]
print "test_target"
print test_target
test_data = iris.data[test_idx]
print "test_data"
print test_data

# Train using decision tree classifier

clf = tree.DecisionTreeClassifier()
clf.fit(test_data,test_target)
print test_target

print clf.predict(test_data)

# Graphviz code vo visualize Classifier Tree

# from sklearn.externals.six import StringIO
# import pydot
# dot_data = StringIO()
# tree.export_graphviz(clf,
#                         out_file=dot_data,
#                         feature_names=iris.feature_names,
#                         class_names=iris.target_names,
#                         filled=True, rounded=True,
#                         impurity=False)
#
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iristree.pdf")
