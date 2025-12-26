import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

"""
Load dataset into dataframe.
"""
dataset_path = os.path.abspath('./datasets/raw/bank-full.csv')
df = pd.read_csv(dataset_path)

"""
Feature and target columns definition.
"""
feature_cols = ["age", "job", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]
target_col =  "y"

"""
Convert categorical features to one-hot encoding so KNN receives numeric inputs.
"""
categorical_cols = df[feature_cols].select_dtypes(include="object").columns
x_df = pd.get_dummies(df[feature_cols], columns=categorical_cols)

x = x_df.to_numpy()
y = df[target_col].to_numpy()

"""
KNN configuration params.
"""
knn_n_neighbors=1
knn_weights="distance"
knn_algorithm="auto"
knn_leaf_size=30
knn_p=2
knn_metric="minkowski"
knn_metric_params=None
knn_n_jobs=1

"""
KNN configuration.
"""
knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors, 
                           weights=knn_weights, 
                           algorithm=knn_algorithm, 
                           leaf_size=knn_leaf_size, 
                           p=knn_p, 
                           metric=knn_metric, 
                           metric_params=knn_metric_params, 
                           n_jobs=knn_n_jobs)

"""
Train & Test split.
"""
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=0
)

"""
Fit KNN to data.
"""
knn.fit(X_train, y_train)

"""
Predict on test set.
"""
y_pred = knn.predict(X_test)

"""
Save experiment metadata.
"""
metadata = {
    "name": "knn",
    "params": {
        "n_neighbors": knn_n_neighbors, 
        "weights": knn_weights, 
        "algorithm": knn_algorithm, 
        "leaf_size": knn_leaf_size, 
        "p": knn_p, 
        "metric": knn_metric, 
        "metric_params": knn_metric_params, 
        "n_jobs": knn_n_jobs
    },
    "dataset": {
        "path": dataset_path
    },
    "labels": {
        "features": feature_cols,
        "target": target_col
    },
    "y_test": y_test,
    "y_pred": y_pred,
    "classes": knn.classes_
}
with open(os.path.join("results", "knn", "metadata.json"), "w") as f:
    json.dump(metadata, f, cls=NumpyEncoder)

"""
Save model.
"""
with open(os.path.join("results", "knn", "model.pkl"), "wb") as knnPickle: 
    pickle.dump(knn, knnPickle)  
    knnPickle.close()