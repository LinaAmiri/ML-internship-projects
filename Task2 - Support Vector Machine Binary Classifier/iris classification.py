import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

#load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#linear kernel
svm_linear = SVC(kernel="linear", probability=True, random_state=42)
svm_linear.fit(X_train, y_train)

#RBF kernel
svm_rbf = SVC(kernel="rbf", probability=True, random_state=42)
svm_rbf.fit(X_train, y_train)

#predictions
y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

#reports
print("Linear Kernel Classification Report:\n", classification_report(y_test, y_pred_linear, target_names=target_names))
print("RBF Kernel Classification Report:\n", classification_report(y_test, y_pred_rbf, target_names=target_names))

#Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(confusion_matrix(y_test, y_pred_linear), annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Confusion Matrix - Linear Kernel")

sns.heatmap(confusion_matrix(y_test, y_pred_rbf), annot=True, fmt="d", cmap="Greens", ax=ax[1])
ax[1].set_title("Confusion Matrix - RBF Kernel")
plt.tight_layout()
plt.show()

#binarize labels for multi-class ROC AUC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

#ROC AUC
roc_auc_linear = roc_auc_score(y_test_bin, svm_linear.decision_function(X_test), multi_class="ovr")
roc_auc_rbf = roc_auc_score(y_test_bin, svm_rbf.decision_function(X_test), multi_class="ovr")

print(f"ROC AUC (Linear): {roc_auc_linear:.3f}")
print(f"ROC AUC (RBF): {roc_auc_rbf:.3f}")

import numpy as np
import os

def plot_decision_boundary(model, X, y, title):
    X = X[:, :2]  #only first 2 features for visualization
    model.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", edgecolor="k")
    plt.title(title)
    plt.show()

results_dir = "results"
base_dir = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(base_dir, results_dir)
os.makedirs(results_path, exist_ok=True)

def plot_decision_boundary_and_save(model, X, y, title, filename):
    X = X[:, :2]  #only first 2 features for visualization
    model.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", edgecolor="k")
    plt.title(title)
    plt.savefig(os.path.join(results_path, filename))
    plt.close()

#plot and save with Linear & RBF kernels
plot_decision_boundary_and_save(SVC(kernel="linear"), X_train, y_train, 
    "Decision Boundary - Linear Kernel (2 features)", "decision_boundary_linear.png")
plot_decision_boundary_and_save(SVC(kernel="rbf"), X_train, y_train, 
    "Decision Boundary - RBF Kernel (2 features)", "decision_boundary_rbf.png")

