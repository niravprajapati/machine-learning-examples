from sklearn import tree
# smooth = 0, bumpy = 1. Apple is lower in weight and smooth, orange is bit bumpy
features = [[140,0],[132,0],[150,1],[150,0],[142,0],[156,1],[135,0]]
# 0 = apple, 1 = orange
labels = [0,0,1,0,0,1,0]
classifier = tree.DecisionTreeClassifier()
classifier.fit(features,labels)
v = classifier.predict([[167,1]])
if(v == 1):
    print("orange")
else:
    print ("apple")
