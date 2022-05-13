import numpy as np
import pandas
from sklearn.model_selection import StratifiedKFold
from get_data import get_data
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.random import sample_without_replacement
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from copy import deepcopy
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical
from keras.utils import plot_model


def normalizer(x_train , x_test):

	minmax = MinMaxScaler()
	minmax.fit(x_train)
	print("data normalized!")
	return minmax.transform(x_train), minmax.transform(x_test)


def pre_process(x_train , x_test):

	pca = PCA (n_components = 0.99 ,  svd_solver = "full")
	pca.fit(x_train)
	return pca.inverse_transform(pca.transform(x_train)) , pca.inverse_transform(pca.transform(x_test))


def predict(x, clfs):
	res = []
	for clf in clfs:
		res.append(clf.predict_proba(x))

	res = np.concatenate(res , axis = 1)
	return res


def sampling(x,y,subsample_size=1):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys


def fit(x_train, y_train, x_test, y_test, clfs):
	res = []
	for clf in clfs:
		clf.fit(x_train,y_train)
		y = clf.predict(x_test)


def classifying(x_train, y_train, x_test, y_test):
	Allclfs = [
		GaussianNB(),

		LogisticRegression(penalty='l2', C=10, solver='lbfgs'),
		LogisticRegression(penalty='l2', C=100.0, solver='lbfgs'),

		KNeighborsClassifier(n_neighbors=4),
		KNeighborsClassifier(n_neighbors=32),
		
		
		RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=10),
		RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=5),
		
		AdaBoostClassifier(n_estimators=80, learning_rate=0.1),

		GradientBoostingClassifier(learning_rate=0.1, n_estimators=50),
		GradientBoostingClassifier(learning_rate=0.01, n_estimators=80),

		SVC(C=1.0, gamma='auto', probability=True),
		SVC(C=10.0, gamma='auto', probability=True),

	]
	Allclfs += deepcopy(Allclfs)
	fit(x_train, y_train, x_test, y_test, Allclfs)
	x_train = np.concatenate((x_train, predict(x_train, Allclfs)), axis=1)
	x_test = np.concatenate((x_test, predict(x_test, Allclfs)), axis=1)
	print("first classifiers trained")
	return x_train, x_test


def CNN(x_train  , y_train , x_test , y_test):
	number_of_classes = len(np.unique(y_train))
	y_train = to_categorical(y_train, num_classes=number_of_classes)
	y_test = to_categorical(y_test, num_classes=number_of_classes)

	clf = Sequential([
	    Dense(1024, activation = 'relu', input_shape=(len(x_train[0]),)),
	    Dropout(0.50),
	    Dense(512, activation = 'relu'),
	    Dropout(0.50),
	    Dense(128, activation = 'relu'),
	    Dense(32, activation = 'relu'),
	    Dense(number_of_classes, activation = 'sigmoid')
	])

	clf.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

	clf.fit(x_train , y_train, epochs=100, verbose = 2, batch_size = 512
		, validation_split= 1/10, callbacks = [EarlyStopping(patience = 25)])
	print("nueral network trained")
	return clf.evaluate(x_test , y_test)[1]


def main():
	x_data , y_data , y_data_labels= get_data()
	


	total_accuracy = 0
	cnt = 1
	Nfolds = 10  # this can be changed

	kf = StratifiedKFold(n_splits=Nfolds , shuffle  = True)
	
	for train_index, test_index in kf.split(x_data, y_data):

		X_train, X_test = x_data[train_index], x_data[test_index]
		y_train, y_test = y_data[train_index], y_data[test_index]
		X_train , X_test = pre_process(X_train , X_test)
		X_train, X_test = normalizer(X_train, X_test)
		X_train, X_test = classifying(X_train, y_train, X_test, y_test)
		X_train , X_test = pre_process(X_train , X_test)
		X_train, X_test = normalizer(X_train, X_test)

		accuracy = CNN(X_train, y_train, X_test, y_test)
		
		total_accuracy += accuracy
		print("Accuracy of " + str(cnt) + " = " , accuracy )
		cnt+=1

	total_accuracy/=Nfolds

	print("final accuracy  = ", total_accuracy)

if __name__ == '__main__':
	main()
