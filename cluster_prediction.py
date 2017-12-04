import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

def predict_NB_Bernoulli(X, Y):
	X_dev, X_test, y_dev, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
	X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.15/0.85, random_state=42)
	clf = BernoulliNB()
	clf.fit(X_train, y_train)
	print("Accuracy: " + str(clf.score(X_val, y_val)))
	prob_X = clf.predict_proba(X_val)
	prob_score = 0
	for i in range(len(y_val)):
		prob_score += prob_X[i][y_val[i]]
	print("Average prob for correct classes: " + str(prob_score/len(y_val)))
	return clf.score(X_val, y_val)


