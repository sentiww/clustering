import matplotlib.pyplot as plt
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description="Visualization utility.")
parser.add_argument("--input", help="Root directory of experiment result.")
args = parser.parse_args()

df = pd.read_csv('./datasets/raw/bank-full.csv')

with open(os.path.join(args.input, "metadata.json")) as f:
    metadata = json.load(f)

    classes = metadata["classes"]
    feature_cols = metadata["labels"]["features"]
    target_col =  metadata["labels"]["target"]

    categorical_cols = df[feature_cols].select_dtypes(include="object").columns
    x_df = pd.get_dummies(df[feature_cols], columns=categorical_cols)

    x = x_df.to_numpy()
    y = df[target_col].to_numpy()

    pca = PCA(n_components=2, random_state=0)
    embedding = pca.fit_transform(x)

    label_cats = pd.Categorical(y)
    label_codes = label_cats.codes

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=label_codes, cmap="tab10", s=10)
    plt.title("PCA projection colored by label")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    legend_handles, _ = scatter.legend_elements()
    plt.legend(legend_handles, label_cats.categories, title="Label", fontsize="small", loc="best")
    plt.tight_layout()
    plt.show()