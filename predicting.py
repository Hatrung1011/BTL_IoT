### Data Handline
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Importing the dataset
df = pd.read_csv("D:/Workspace/BTL_IoT/heart.csv")

df
df.shape

df.head()

df.describe()

# Check data null
df.isnull().sum()

df['target'].value_counts()

X = df.drop(columns = 'target', axis = 1)
Y = df['target']

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y,random_state = 2)


# Training the Decision Tree Classifier model on the Training set
from sklearn import tree

model = tree.DecisionTreeClassifier(random_state=0, max_depth=4)
model = model.fit(X_train, Y_train)
tree.plot_tree(model)

from sklearn.metrics import accuracy_score, confusion_matrix
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

train_data_accuracy

from sklearn.metrics import accuracy_score
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

test_data_accuracy

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def draw_cf_matrix(label, pred):

  cf_matrix = confusion_matrix(label, pred)

  ax = plt.subplot()

  sns.heatmap(cf_matrix, annot=True, ax = ax, fmt='g', cmap='Greens')

  ax.set_xlabel('Predicted labels')

  ax.set_ylabel('True labels')

  ax.set_title('Confusion Matrix')

  ax.xaxis.set_ticklabels([0, 1])

  ax.yaxis.set_ticklabels([0, 1])

  cf_matrix

draw_cf_matrix(Y_test, X_test_prediction)

print(classification_report(Y_test, X_test_prediction))

# Predicting system
data = (62,0,0,138,294,1,1,106,0,1.9,1,3,2)
data_array = np.asarray(data)
data_reshape = data_array.reshape(1, -1)
data_standard = scaler.transform(data_reshape)
prediction = model.predict(data_standard)
if(prediction[0] == 1):
    print('Heart Disease')
else:
    print('No Heart Disease')
