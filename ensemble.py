import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    cohen_kappa_score,
)

from utils.ensemble import MajorityVoting
from utils.utils import load_data_file

MODE = "Binary"

if __name__ == "__main__":
    # Load the model
    run_ids = [
        "d14d502cca984bcb8b0e9f66deec8cd2",
        "73f0abbe48dc4ca19cdb9b74a1826521",
        "6daf9a790a8d471b80a16ca45b8b5be3",
        "2af79e5e3bb140a190c8bd18a67bdeaa",
    ]
    if MODE == "Binary":
        classes = np.loadtxt("./datasets/Binary/classes.txt", dtype=str)
        names, labels = load_data_file("./datasets/Binary/val.txt")
    else:
        classes = np.loadtxt("./datasets/Multiclass/classes.txt", dtype=str)
        names, labels = load_data_file("./datasets/Multiclass/val.txt")

    mv = MajorityVoting(run_ids=run_ids, tta=True)
    predicts = mv.predict()
    # Calculate the accuracy
    acc = accuracy_score(labels, predicts)
    print(f"Ensemble Accuracy: {acc:.4f}")

    # Classification report
    print(classification_report(labels, predicts, target_names=classes))

    # AUC
    auc_score = roc_auc_score(predicts, labels)
    print(f"AUC: {auc_score:.4f}")

    # Cohen's Kappa
    kappa = cohen_kappa_score(labels, predicts)
    print(f"Cohen's Kappa: {kappa:.4f}")
