import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df.describe()
df.info()

sns.jointplot(x='col1', y='col2', data=df)
sns.pairplot(data) # hue='col3', palette='bwr'
sns.distplot(data.col1, bins=50)
plt.scatter(actuals, predictions)

# Preprocessing
df.drop(['Col7', 'Col8', 'Col9'], axis = 1, inplace = True)

# Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop(['TARGET CLASS'], axis = 1))
df_scaled = scaler.transform(df.drop(['TARGET CLASS'], axis = 1))

# One Hot Encoding (get_dummies)
final_data = pd.get_dummies(loans, columns=['col1', 'col2'], drop_first=True)



### Train Test Split
from sklearn.model_selection import train_test_split
y = df.pop('Target')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# Linear Regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
# Coefficients = "lm.coef_"
predictions = lm.predict(X_test)
plt.scatter(actuals, predictions)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clas = KNeighborsClassifier(n_neighbors = 1)

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)

# KMeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private',axis=1))

# KMeans with Silhoutte score
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(df)
    centers = clusterer.cluster_centers_

    score = silhouette_score(df, preds)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))


### Model Evaluation
from sklearn.metrics import classification_report, mse, mae
print(classification_report(y_test, predictions))
