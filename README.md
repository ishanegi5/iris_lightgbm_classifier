# Iris Flower Classification using LightGBM 🌸

This project demonstrates the use of **LightGBM (Light Gradient Boosting Machine)** for multiclass classification on the classic **Iris dataset**.  
The model classifies iris flowers into three species: *Setosa*, *Versicolor*, and *Virginica*.

---

## 📌 Project Overview
- Implemented a **LightGBM classifier** for multiclass classification.
- Dataset: [Iris Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset)
- Achieved **100% accuracy** on test data.
- Evaluated using **accuracy score, confusion matrix, and classification report**.

---

## 📂 Dataset
The dataset consists of 150 samples of iris flowers, each described by:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)  
Target: 3 classes (Setosa, Versicolor, Virginica)

---

## ⚙️ Model
```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=50,
    max_depth=5,
    num_leaves=3,
    objective='multiclass',
    num_class=3,
    random_state=42
)

model.fit(X_train, y_train)
📊 Results
Accuracy: 100%

Confusion Matrix:


[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
Classification Report:

Class	Precision	Recall	F1-score	Support
0	1.00	1.00	1.00	10
1	1.00	1.00	1.00	9
2	1.00	1.00	1.00	11

🔑 Key Learnings
LightGBM handles multiclass classification efficiently.

Feature importance can be visualized to understand model decisions.

Hyperparameter tuning (e.g., num_leaves, max_depth) improves performance.

🚀 How to Run
Clone this repo:


git clone https://github.com/ishanegi5/iris_lightgbm_classifier.git
cd iris_lightgbm_classifier
Install requirements:


pip install -r requirements.txt
Run the Jupyter notebook or Python script.

📦 Requirements
Python 3.x

scikit-learn

lightgbm

pandas

matplotlib (optional, for feature importance plot)

Install all with:

pip install lightgbm scikit-learn pandas matplotlib

👩‍💻 Author
Isha Negi
