# 🌸 Task 2 - Support Vector Machine (SVM) Classifier (Iris Dataset)

This project was developed as part of my **Codveda Internship**.  
The goal was to implement a **Support Vector Machine (SVM)** classifier using different kernels on the classic **Iris dataset**, and compare their performance.

---

## 📌 Objectives
- Train an SVM model on the Iris dataset.  
- Compare the performance of **Linear kernel** vs **RBF kernel**.  
- Visualize **confusion matrices** and **decision boundaries**.  
- Evaluate using accuracy, precision, recall, F1-score, and ROC AUC.  

---

## 📂 Dataset
- **Dataset**: Iris dataset from scikit-learn (150 samples, 3 classes).  
- **Classes**:  
  - `setosa`  
  - `versicolor`  
  - `virginica`  
- **Features**: Sepal length, sepal width, petal length, petal width.  

---

## ⚙️ Tools & Libraries
- Python 🐍  
- [Scikit-learn](https://scikit-learn.org/stable/)  
- [Pandas](https://pandas.pydata.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [Seaborn](https://seaborn.pydata.org/)  

---

## 🚀 Results

### 🔹 Linear Kernel
- **Accuracy:** 100%  
- **ROC AUC:** 1.000  
- Perfect classification across all 3 classes.  

### 🔹 RBF Kernel
- **Accuracy:** 97%  
- **ROC AUC:** 0.998  
- Minor misclassification between *versicolor* and *virginica*.  

---

## 📊 Visualizations

### Confusion Matrices
Linear vs RBF kernel classification performance:

![Confusion Matrix - Linear](figures/confusion_matrix_linear.png)  
![Confusion Matrix - RBF](figures/confusion_matrix_rbf.png)  

### Decision Boundaries
(2D projection using first two features)  

![Decision Boundary - Linear](figures/decision_boundary_linear.png)  
![Decision Boundary - RBF](figures/decision_boundary_rbf.png)  

---

## 📈 Insights
- The **Linear kernel SVM** perfectly separates the Iris classes.  
- The **RBF kernel SVM** is almost as good but shows slight overlap between *versicolor* and *virginica*.  
- Both models achieve **near-perfect ROC AUC (>0.99)**, confirming strong classification ability.  

---

## ▶️ How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/LinaAmiri/codveda-internship.git
   cd "Task2 - Support Vector Machine Binary Classifier"
