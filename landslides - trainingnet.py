import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sgd = optimizers.SGD(lr=0.01, clipvalue =0.5)
seed = 35
numpy.random.seed(seed)
dataframe = pandas.read_csv("newtrain.csv", header=1)
dataframe = pandas.DataFrame(dataframe)
dataset = dataframe.values.astype('str')
numpy.nan_to_num(dataset)
#dataset.fillna(0, inplace=True)
X = dataset[:,0:4]
Y = dataset[:,4:5]
#feature_names = ["latitude", "longitude", "setting", "trigger"]
#Z = dataset [:,4]
print (X.shape)
print (Y.shape)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
var_y = np_utils.to_categorical(encoded_Y)
encoder.fit(X)
encoded_X = encoder.transform(X)
var_x = np_utils.to_categorical(encoded_X)

training_features, test_features, training_target, test_target, = train_test_split(x, var_y, test_size = .3, random_state=seed)

#print (x.shape)
#print (var_y.shape)

def modelsize():
	# create model
	model = Sequential()
	model.add(Dense(100, input_dim=24764, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(25, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=modelsize, epochs=200, batch_size=15, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, x, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#predictions = estimator.predict(test_features)


predictions = cross_val_predict(estimator, x, y, cv = kfold)
print (predictions)
print (y.shape)
print (predictions.shape)
new_y = y[:,:1]
#new_y = y.flatten()
#y.reshape(3702,)
#new_y = PCA(n_components=0).fit_transform(y)
#new_pred = PCA(n_components=0).fit_transform(predictions)
print (new_y.shape)
#print (new_pred.shape)
fig, ax = plt.subplots()
ax.scatter(new_y, predictions, edgecolors=(0, 0, 0))
ax.plot([new_y.min(), new_y.max()], [new_y.min(), new_y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

#perm = PermutationImportance(estimator, random_state=1).fit(x,y)
#eli5.show_weights(perm, feature_names = x.columns.tolist())

#conf = confusion_matrix(new_y, predictions)
#print (conf)
