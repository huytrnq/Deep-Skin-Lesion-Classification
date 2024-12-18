import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    cohen_kappa_score,
)

from utils.ensemble import Ensemble
from utils.utils import load_data_file

# DATASET = "Multiclass"
DATASET = "Binary"
TTA = False

if __name__ == "__main__":
    # Load the model
    ## Binary run_ids
    if DATASET == "Binary":
        run_ids = [
            "d14d502cca984bcb8b0e9f66deec8cd2",
            "73f0abbe48dc4ca19cdb9b74a1826521",
            # "c440d5e38b764a32aa66bd623545794e",
            # "1c717a394d854ec8ada7edd7dfe57feb",
            # "3143486234e64ee38f8917e6bffa5de2",
            "aca333832cbf492981651b12b6f27c84",
            "01075d0203854b33b15069fe44ffadcd",
        ]
    else:
        ## Multiclass run_ids
        run_ids = [
            "776b2e8f8853416a9c959b312a5a4611",
            "f62f6e145791420caf0346263e4b14fa",
            "b91974f242e24f8e91ca3bda4b988b92",
        ]
    classes = np.loadtxt(f"./datasets/{DATASET}/classes.txt", dtype=str)
    names, labels = load_data_file(f"./datasets/{DATASET}/val.txt")

    ensemble = Ensemble(
        run_ids=run_ids,
        mode="average",  # Change to the desired mode
        tta=False,
        weights=None,
    )
    predicts = ensemble.predict()
    # Calculate the accuracy
    acc = accuracy_score(labels, predicts)
    print(f"Ensemble Accuracy: {acc:.4f}")

    # Classification report
    print(classification_report(labels, predicts, target_names=classes))

    # Cohen's Kappa
    kappa = cohen_kappa_score(labels, predicts)
    print(f"Cohen's Kappa: {kappa:.4f}")
