# Filename: DecisionTree_scikit_py3.py.py
# Function: Source code to Implement The Decision Tree Algorithm with scikit in Python
# Reference: 
# See book Mastering Machine Learning with Python in Six Steps
#          listing 3-36 p178


# CART on the Bank Note dataset
from random import seed
from random import randrange
from csv import reader

from sklearn import tree
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

# Load a CSV file
def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset


print("Extra apply sklearn functions")
# Test CART on Bank Note dataset
seed(1)
# load and prepare data
filename = 'data_banknote_authentication.csv'
dataset = load_csv(filename)

# transform dataset to X and y
X = []
y = []
for record in dataset:
	#print("record: ", record, " record[0,4]: ", record[0:4], " type(): ", type(record[0:4]), type(record) )
	X.append(list(record[0:4]))
	y = y + list(record[4])

# Check aantallen
print("len(dataset): ", len(dataset))
print("len(X): ", len(X))
print("len(y): ", len(y))

# call sci-learn to create Decision Tree 
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
# split data into train and test
print("OK")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state=0)
clf.fit(X_train, y_train)
## generate evaluation metrics
#print("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train)))
#print("Train - Confusion matrix :",metrics.confusion_matrix(y_train, clf.predict(X_train)))

tree.export_graphviz(clf, out_file='tree.dot')
import pydot
# hieronder gaat fout
# zoek voorbeeld code print de decision tree
from sklearn.externals.six import StringIO
#out_data = StringIO()
'''
tree.export_graphviz(clf, out_file=out_data,
                     feature_names='test waarde aanpassen',
                     class_names=clf.classes_.astype(int).astype(str),
                     filled=True, rounded=True,
                     special_characters=True,
                     node_ids=1,)
graph = pydot.graph_from_dot_data(out_data.getvalue())
graph[0].write_pdf("boom.pdf") # save to pdf
'''

