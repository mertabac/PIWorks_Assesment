import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
# Confusion matrix defining
def plot_conf_mat(y_test, y_preds, title):

    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
# Reading the data
data = pd.read_csv("dataset.csv")

# Converting NaN values to median value because it's accuracy is better than mean value, i tried it first
data.fillna(data.median(), inplace=True)

# Column selecting
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#KNN
# KNN range definition
n_neighbors = range(2, 21)

# Creating new dataframe for easy reading experience
results_knn = pd.DataFrame(columns=['n_neighbors', 'accuracy'])

# Her komşu sayısı için bir KNN modeli oluşturma ve doğruluğunu hesaplama
for k in n_neighbors:
    # Split data into train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
    
    # Model training
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Calculating accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Appending results
    results_knn = results_knn.append({'n_neighbors': k, 'accuracy': accuracy}, ignore_index=True)

# Plotting confusion matrix
print(plot_conf_mat(y_test, y_pred, 'KNN'))

# Linear Regression; Score was terrible so i turned this model into a comment
"""model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)


print("R2 score:", r2)"""

# Decision Tree
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print("Decision Tree method accuracy score: {:.2f}".format(score))

# Random Forest
results_random_forest = []
for n in [100, 200, 300, 400, 500]:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    results_random_forest.append([n, score])

results_random_forest_df = pd.DataFrame(results_random_forest, columns=["n_estimators", "accuracy"])
print(plot_conf_mat(y_test, y_pred,'Random Forest'))

# Visualization on best models
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(results_random_forest_df['n_estimators'], results_random_forest_df['accuracy'], 'bo-', label='Accuracy')
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('Accuracy')
ax1.set_title('Random Forest Classifier')
ax1.legend()

ax2.plot(results_knn['n_neighbors'], results_knn['accuracy'], 'ro-', label='Accuracy')
ax2.set_xlabel('n_neighbors')
ax2.set_ylabel('Accuracy')
ax2.set_title('K-Nearest Neighbors Classifier')
ax2.legend()

plt.show()

"""
This code shows us best fitting model is a 200 estimator random forest classifier
"""


#best function
def best_fitting_model(X_train, y_train, X_test, y_test):
    magic = RandomForestClassifier(n_estimators=200)
    magic.fit(X_train, y_train)
    accuracy = magic.score(X_test, y_test)
    return magic, accuracy

