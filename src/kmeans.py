import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
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

work_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

"""
Load dataset into dataframe.
"""
dataset_path = os.path.join(work_dir, "datasets", "raw", "bank-full.csv")
df = pd.read_csv(dataset_path)

"""
Feature and target columns definition.
"""
feature_cols = ["age", "job", "education", "default", "balance", "housing", "loan", "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome"]
target_col =  "y"

"""
Convert categorical features to one-hot encoding so KMeans receives numeric inputs.
"""
categorical_cols = df[feature_cols].select_dtypes(include="object").columns
x_df = pd.get_dummies(df[feature_cols], columns=categorical_cols)

x = x_df.to_numpy()
y = df[target_col].to_numpy()

"""
KMeans configuration params.
"""
kmeans_n_clusters=1000
kmeans_init='k-means++'
kmeans_n_init='auto'
kmeans_max_iter=1000
kmeans_tol=0.0001
kmeans_verbose=0
kmeans_random_state=None
kmeans_copy_x=True
kmeans_algorithm='lloyd'

"""
KMeans configuration.
"""
kmeans = KMeans(n_clusters=kmeans_n_clusters, 
                init=kmeans_init, 
                n_init=kmeans_n_init, 
                max_iter=kmeans_max_iter, 
                tol=kmeans_tol, 
                verbose=kmeans_verbose, 
                random_state=kmeans_random_state, 
                copy_x=kmeans_copy_x, 
                algorithm=kmeans_algorithm)

"""
Train & Test split.
"""
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=0
)

"""
Fit KMeans to data.
"""
kmeans.fit(X_train)

"""
Map each cluster to the majority class seen in the training set.
"""
train_clusters = kmeans.labels_
cluster_to_class = {}
for cluster_id in range(kmeans_n_clusters):
    cluster_labels = y_train[train_clusters == cluster_id]
    if cluster_labels.size == 0:
        cluster_to_class[cluster_id] = pd.Series(y_train).mode()[0]
    else:
        cluster_to_class[cluster_id] = pd.Series(cluster_labels).mode()[0]

"""
Predict on test set and convert cluster ids to class labels.
"""
cluster_preds = kmeans.predict(X_test)
y_pred = np.array([cluster_to_class[c] for c in cluster_preds])

"""
Save experiment metadata.
"""
metadata = {
    "name": "kmeans",
    "params": {
        "n_clusters": kmeans_n_clusters, 
        "init": kmeans_init, 
        "n_init": kmeans_n_init, 
        "max_iter": kmeans_max_iter, 
        "tol": kmeans_tol, 
        "verbose": kmeans_verbose, 
        "random_state": kmeans_random_state, 
        "copy_x": kmeans_copy_x, 
        "algorithm": kmeans_algorithm
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
    "classes": ["no", "yes"],
    "cluster_to_class": cluster_to_class
}
with open(os.path.join(work_dir, "results", "kmeans", "metadata.json"), "w") as f:
    json.dump(metadata, f, cls=NumpyEncoder)

"""
Save model.
"""
with open(os.path.join(work_dir, "results", "kmeans", "model.pkl"), "wb") as kmeansPickle: 
    pickle.dump(kmeans, kmeansPickle)  
    kmeansPickle.close()
