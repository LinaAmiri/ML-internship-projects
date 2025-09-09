# rf_churn.py
# Codveda Internship - Task 1: Random Forest Classifier
# Dataset: Churn Prediction (split 80% train / 20% test provided)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import os

RANDOM_STATE = 42
#load data
train_df = pd.read_csv("C:/Users/2024/Desktop/ML internship projects/Task1 - Random Forest Classifier\Churn Prdiction Dataset\Churn Prdiction Data/churn-bigml-80.csv")
test_df  = pd.read_csv("C:/Users/2024/Desktop/ML internship projects/Task1 - Random Forest Classifier\Churn Prdiction Dataset\Churn Prdiction Data/churn-bigml-20.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

#preprocessing
#drop customerID if present
for df in [train_df, test_df]:
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

#concatenate train+test to ensure same encoding
full = pd.concat([train_df, test_df])

#oe-hot encode categorical (exclude target 'Churn')
cat_cols = full.select_dtypes(include=['object','category']).columns.tolist()
if 'Churn' in cat_cols:
    cat_cols.remove('Churn')

full = pd.get_dummies(full, columns=cat_cols, drop_first=True)

#split back into train/test
train_df = full.iloc[:len(train_df)].copy()
test_df  = full.iloc[len(train_df):].copy()

#target variable
y_train = train_df['Churn'].map({'Yes':1,'No':0}) if train_df['Churn'].dtype=='object' else train_df['Churn']
X_train = train_df.drop(columns=['Churn'])

y_test = test_df['Churn'].map({'Yes':1,'No':0}) if test_df['Churn'].dtype=='object' else test_df['Churn']
X_test = test_df.drop(columns=['Churn'])

#fill missing numeric values
X_train = X_train.fillna(X_train.median())
X_test  = X_test.fillna(X_test.median())

# Optional: scale numeric (RF doesn’t need it, but keeps consistency)
num_cols = X_train.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

print("Train ready:", X_train.shape, "Test ready:", X_test.shape)

#train & tune Random Forest

param_grid = {
    'n_estimators': [100, 250],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid = GridSearchCV(rf, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
best_rf = grid.best_estimator_

#Cross-validation on train-
y_pred_cv = cross_val_predict(best_rf, X_train, y_train, cv=cv)
print("\nCross-validation results (Train set):")
print(classification_report(y_train, y_pred_cv, digits=4))

#final evaluation on test set
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:,1]

print("\nFinal Evaluation on Test Set:")
print(classification_report(y_test, y_pred, digits=4))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

os.makedirs("results", exist_ok=True)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("Task1 - Random Forest Classifier/results/confusion_matrix.png")
plt.close()


#feature importance
print("\nComputing permutation importances...")
perm = permutation_importance(best_rf, X_test, y_test,
                              n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
feat_imp = pd.DataFrame({
    'feature': X_test.columns,
    'importance_mean': perm.importances_mean,
    'importance_std': perm.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nTop 10 Features by Importance:")
print(feat_imp.head(10))


top = feat_imp.head(15).iloc[::-1]
plt.figure(figsize=(8,6))
plt.barh(top['feature'], top['importance_mean'])
plt.xlabel("Permutation Importance")
plt.title("Top Features - Random Forest")
plt.tight_layout()
plt.savefig("Task1 - Random Forest Classifier/results/feature_importance.png")
plt.close()
