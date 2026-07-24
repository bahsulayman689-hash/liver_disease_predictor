import random 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, auc, precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

from sklearn.metrics import precision_score, recall_score, f1_score

liver_disease = pd.read_csv("C:\\Users\\Sulayman Bah\\Desktop\\model\\Liver_disease_data.csv")
print(liver_disease.head())
print(liver_disease.shape)
print(liver_disease.info())
print(liver_disease.isnull().sum())

X = liver_disease.drop('Diagnosis', axis=1)
print(X)
print(X.dtypes)
y = liver_disease["Diagnosis"]
print(y)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_log = model.predict(X_test)
Y_testing_Accu = accuracy_score(y_test, y_pred_log)
class_report = classification_report(y_test, y_pred_log)
confus_report = confusion_matrix(y_test, y_pred_log)
score = cross_val_score(model,
                        X, y, cv=3,
                        )
predict = cross_val_predict(model, X, y, cv=3, method='predict')
print(f"the accuracy of the model {Y_testing_Accu}")
print(f"class_report {class_report}")
print(f"confusion_met {confus_report}")
print(f"the score {score}")
print(f"predict of the mode using {predict}")
test = np.argmax(score)
test_1 = np.argmax(predict)
pre_recall = precision_score(y_test, y_pred_log)
recall_scor = recall_score(y_test, y_pred_log)
print(pre_recall)
print(recall_scor)
print(f"test {test}")
print(f"thest of the pre {test_1}")
print("="*70, '\n')
joblib.dump(model, "liver_disease_model.pkl")
model = joblib.load("liver_disease_model.pkl")

joblib.dump(scaler, "scaler.pkl")
scaler = joblib.load("scaler.pkl")
print('\n', "="*70)
print(X.columns)
print(len(X.columns))