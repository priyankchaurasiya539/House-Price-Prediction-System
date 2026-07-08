import pandas as pd
import joblib 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV , StratifiedKFold
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

#load the saved models 

X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")


#Standardization

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Apply Logistic Regression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled , y_train)

#Predictions

train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

print("Train Accuracy :", round(accuracy_score(y_train, train_pred), 4))
print("Test Accuracy  :", round(accuracy_score(y_test, test_pred), 4))

#Now applyling grid search cv

penalty = ['l1' , 'l2' , 'elasticnet']
c_values = [100 , 10 , 1 , 0.1 , 0.01 ]
solver = ['newton-cg' , 'saga' , 'liblinear' , 'lbfgs' , 'sag']
params = dict(penalty=penalty , C= c_values , solver = solver)

#Apply GridSearchCV

cv = StratifiedKFold()
grid = GridSearchCV(
    estimator=model,
    param_grid=params,
    scoring="accuracy",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train_scaled , y_train)
print("Best Params : " , grid.best_params_)
print("Accuracy score : " , grid.best_score_)


# Best Model
best_model = LogisticRegression(
    C=1, 
    penalty='l2', 
    solver='sag',
    max_iter=1000
)
best_model.fit(X_train_scaled , y_train)

#Final evaluation

pred = best_model.predict(X_test_scaled)

print("Final Test Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

#Save the models

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("All necessary files saved for future data testing!")