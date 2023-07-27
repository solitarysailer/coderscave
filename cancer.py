import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

breast_cancer_data = load_breast_cancer()

breast_cancer_data.feature_names

breast_cancer_data.data

breast_cancer_data.target_names

breast_cancer_data.target

df = pd.DataFrame(data=breast_cancer_data.data,columns=breast_cancer_data.feature_names)
df.head()

df_target = pd.DataFrame(data=breast_cancer_data.target, columns=['target'])
df_target.head()

df = pd.merge(df, df_target, left_index=True, right_index=True)
df.head()

df.shape

df.info()

df.describe()

df.isnull().sum()

sns.heatmap(df.isnull(), cbar=False)

plt.figure(figsize=(20,8))
sns.heatmap(df.corr(),annot=True)

corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
to_drop
sns.pairplot(df[['mean radius','mean perimeter','mean area','perimeter error','area error','worst radius','worst peri
                 
df.drop(columns=to_drop,inplace=True)
df.shape

def find_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (series < lower_bound) | (series > upper_bound)

# Step 2: Calculate outliers for each column
outliers_per_column = df.apply(find_outliers_iqr)
# Step 3: Identify columns with outliers
columns_with_outliers = outliers_per_column.any()
columns_with_outliers
# # Step 4: Display columns with outliers
print("Columns with outliers:")
print(df.columns[columns_with_outliers].tolist())

for col in df.columns[columns_with_outliers].tolist():
    df.boxplot(by ='target', column =[col], grid = False)
    
    for col in df.columns[columns_with_outliers].tolist():
    sns.distplot(df[col])
    plt.show()
    
    X = df.drop(columns=['target'])
X.shape

y = df['target']
y

min_max_scaler = MinMaxScaler(feature_range =(0, 1))
scaled_X= min_max_scaler.fit_transform(X)
scaled_X

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

X_train.shape

X_test.shape

y_train.shape

y_test.shape

reg = LogisticRegression()
reg.fit(X_train,y_train)

y_pred1 =  reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred1)
accuracy

cm = confusion_matrix(y_test, y_pred1)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

print(classification_report(y_test,y_pred1))

tree=DecisionTreeClassifier()
tree.fit(X_train, y_train)

y_pred2 =  tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred2)
accuracy

cm = confusion_matrix(y_test, y_pred2)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

print(classification_report(y_test,y_pred2))

svm=SVC()
svm.fit(X_train,y_train)

y_pred3 = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred3)
accuracy

cm = confusion_matrix(y_test, y_pred3)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

print(classification_report(y_test,y_pred3))

rforest = RandomForestClassifier(n_estimators=10)
rforest.fit(X_train,y_train)

y_pred4 = rforest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred4)
accuracy

cm = confusion_matrix(y_test, y_pred4)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

print(classification_report(y_test,y_pred4))

knn= KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)

y_pred5 = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred5)
accuracy

cm = confusion_matrix(y_test, y_pred5)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

print(classification_report(y_test,y_pred5))
                 