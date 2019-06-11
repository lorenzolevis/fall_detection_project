#import moduli necessari per l'esecuzione
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#importazione del file CSV
df = pd.read_csv('falldetection.csv')

#cancellazione eventuali dati nulli tramite metodo dropna() 
#implace = True per cancellare direttamente i valori nulli e tirornare None: cio 
#permette di non dover salvare il risultato in una variabile nuova ma eseguirlo direttamente sul dataFrame
len_df = len(df)
df.dropna(inplace=True) 
len_after_df = len(df)

if len(df) == len(df.dropna()):
    print("Non sono stasti trovati dati nulli")
else:
    print("Sono stati trovati " + str(len_df - len_after_df) + " nulli")

#import moduli necessari per l'esecuzione
#scalare i dati su dati reali.
#massimo HR registrato in un essere umano è pari a 208 BPM: applico funzione MinMixScaler con un range da 33 a 208
#la funzione richiede un Numpy array e quindi converto prima la colonna df['HR']
hr_data = df['HR'].astype(float).to_numpy().reshape(-1, 1)
min_max_scaler = MinMaxScaler(copy=True, feature_range=(33, 208))
hr_data_minmax = min_max_scaler.fit_transform(hr_data)
df['HR'] = hr_data_minmax

#diamo un nome reale alle classi e 
fd = df.replace({'ACTIVITY':{0:'Standing', 1:'Walking', 2:'Sitting', 3:'Falling', 4:'Cramps', 5:'Running'}})
print(fd.head(25))

#separazione dataset tra classi e features
target = df['ACTIVITY']
data = df.drop(['ACTIVITY'], axis=1)

test_df = data[['TIME','SL','EEG','BP','HR','CIRCLUATION']]
corr = test_df.corr()

f,ax = plt.subplots(figsize=(12,9))
sns.heatmap(corr, cmap='inferno', annot=True)
plt.show()

#creazione dei modelli di classificazione
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.05)

#conversione esplicita dei file per lo scaler
x_train = x_train.astype(float)
x_test = x_test.astype(float)

#scaling dei dati per migliorare le performance dei vari classificatori
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 1 - SVM - Kernel RBF
model = svm.SVC(kernel='rbf', gamma='auto')
model.fit(x_train, y_train)
model_predictions = model.predict(x_test)

print("SVM accuracy score: ", accuracy_score(y_test, model_predictions))

# 2 - K-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
knn_predictions = knn.predict(x_test)

print("K neighbors perecision: ", accuracy_score(y_test, knn_predictions))

# 3 - Naive Bayes
native_bayes = GaussianNB()
native_bayes.fit(x_train, y_train)
native_bayes_predictions = native_bayes.predict(x_test)

print("native bayes perecision: ", accuracy_score(y_test, native_bayes_predictions))

# - 3 Decision tree Classifier
decisionTree = DecisionTreeClassifier(max_depth=20, random_state=2)
decisionTree.fit(x_train, y_train)
decisionTree_prediction = decisionTree.predict(x_test)

print("DecisionTree precision: ", accuracy_score(y_test, decisionTree_prediction))

#4 - Random Forest Classifier
randomForest = RandomForestClassifier(n_estimators=80, max_depth=20, random_state=2)
randomForest.fit(x_train, y_train)
randomForest_prediction = randomForest.predict(x_test)

print("RandomForest precision: ", accuracy_score(y_test, randomForest_prediction))

# dato che il random forest classifier si è rivelato il migliore andiamo ad analizzare in dettaglio le feature
# che apportano il maggio contributo e un grafico rappresentativo
importances = randomForest.feature_importances_
std = np.std([tree.feature_importances_ for tree in randomForest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],color="b", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), indices)
plt.xlim([-1, x_train.shape[1]])
plt.show()
