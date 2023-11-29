import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('Brain Tumor.csv')

x, y = data.drop(['Image', 'Class'], axis=1), data['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

mx = MinMaxScaler()
mx.fit_transform(x_train)

x_train, x_test = mx.transform(x_train), mx.transform(x_test)

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
'''
clf = LogisticRegression()
clf.fit(x_train, y_train)

pickle.dump(clf, open("mimg.md", 'wb'))
'''

clf = pickle.load(open('nimg.md', 'rb'))
predicted = clf.predict(x_test)

print(metrics.accuracy_score(y_test, predicted)*100)
