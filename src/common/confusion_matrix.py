import matplotlib.pyplot as plt
import argparse
import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

parser = argparse.ArgumentParser(description="Confusion matrix utility.")
parser.add_argument("--input", help="Root directory of experiment result.")
args = parser.parse_args()

with open(os.path.join(args.input, "metadata.json")) as f:
    metadata = json.load(f)

    classes = metadata["classes"]

    y_test = metadata["y_test"]
    y_pred = metadata["y_pred"]

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print("Confusion matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title("KNN Confusion Matrix")
    plt.tight_layout()
    plt.show()