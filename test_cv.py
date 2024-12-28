import os
import argparse
import dagshub
import mlflow
import mlflow.pytorch
import mlflow
import numpy as np
from sklearn.metrics import cohen_kappa_score

import torch
from test import test, load_model_and_config
from utils.utils import export_predictions


def arg_parser():
    """Arg parser

    Returns:
        argparse.Namespace: command line
    """
    parser = argparse.ArgumentParser(
        description="Skin Lesion Classification Experiment"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count()//2, help="Number of workers"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/huy/datasets/",
        help="Path to data directory",
    )
    parser.add_argument(
        "--num_tta",
        type=int,
        default=10,
        help="Number of TTA iterations",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use Test Time Augmentation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="binary",
        help="Dataset type (binary/multiclass)",
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="aca333832cbf492981651b12b6f27c84",
        help="Run ID for loading the model",
    )
    parser.add_argument(
        "--generate_predictions",
        action="store_true",
        help="Use Test Time Augmentation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    # Constants
    RUN_ID = args.run_id
    ARTIFACT_PATH = "config/config.json"
    BATCH_SIZE = args.batch_size if not args.tta else 1
    WORKERS = args.workers
    DEVICE = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Warm-up settings
    # Dataset type
    DATASET = "Binary" if args.dataset.lower() == "binary" else "Multiclass"
    # Classes
    CLASSES = (
        ["nevus", "melanoma", "others"]
        if DATASET == "Multiclass"
        else ["nevous", "others"]
    )

    dagshub.init(
        repo_owner="huytrnq", repo_name="Deep-Skin-Lesion-Classification", mlflow=True
    )

    #################### Test the model with k-folds ####################
    with mlflow.start_run(run_id=RUN_ID):
        fold_test_predictions = []
        fold_test_without_labels = []
        # Load k-folds models
        for fold in range(args.nfolds):
            print(f"Fold {fold + 1}/{args.nfolds}")
            model, config = load_model_and_config(RUN_ID, ARTIFACT_PATH, f"skin_lesion_model_fold_{fold}", DEVICE)
            model = model.to(DEVICE)

            #################### Test the model with k-folds on validation set ####################
            test_acc, kappa_score, prediction_probs, test_labels = test(
                model=model,
                config=config,
                data_file=f"datasets/{DATASET}/val.txt",
                data_root=os.path.join(args.data_root, DATASET),
                batch_size=BATCH_SIZE,
                num_workers=WORKERS,
                device=DEVICE,
                mode="val",
                tta=False if not args.tta else True,
                num_tta=args.num_tta,
                log_kappa=True,
                inference=False,
            )
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Kappa Score: {kappa_score:.4f}")

            fold_test_predictions.append(prediction_probs)

            ############ Generate predictions for the test set ############
            if args.generate_predictions:
                test_prediction_probs = test(
                    model=model,
                    config=config,
                    data_file=f"datasets/{DATASET}/test.txt",
                    data_root=os.path.join(args.data_root, DATASET),
                    batch_size=BATCH_SIZE,
                    num_workers=WORKERS,
                    device=DEVICE,
                    mode="test",
                    tta=False if not args.tta else True,
                    num_tta=args.num_tta,
                    log_kappa=True,
                    inference=True,
                )
                
                fold_test_without_labels.append(test_prediction_probs)

        ### Test Evaluation
        # Average the predictions
        prediction_probs = np.mean(fold_test_predictions, axis=0)
        # Class predictions
        predictions = np.argmax(prediction_probs, axis=1)
        # Test accuracy
        test_acc = np.mean(predictions == test_labels)
        # Cohen's Kappa
        kappa_score = cohen_kappa_score(predictions, test_labels)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Kappa Score: {kappa_score:.4f}")
        # Log metrics
        mlflow.log_metric("test_accuracy" if not args.tta else "test_accuracy_tta", test_acc)
        mlflow.log_metric("kappa_score" if not args.tta else "kappa_score_tta", kappa_score)
        # Export predictions
        export_predictions(prediction_probs, f"results/{DATASET}/prediction_probs.npy" if not args.tta else f"results/{DATASET}/tta_prediction_probs.npy")
        ## Log predictions to artifacts
        mlflow.log_artifact(
            f"results/{DATASET}/prediction_probs.npy" if not args.tta else f"results/{DATASET}/tta_prediction_probs.npy",
        )
        
        ### Generate predictions for the test set
        if not args.generate_predictions:
            test_prediction_probs = np.mean(fold_test_without_labels, axis=0)
            # Class predictions
            test_predictions = np.argmax(test_prediction_probs, axis=1)
            # Export predictions
            export_path = f"results/{DATASET}/test_predictions.npy" if not args.tta else f"results/{DATASET}/test_tta_predictions.npy"
            export_predictions(test_predictions, export_path)
            mlflow.log_artifact(export_path, artifact_path="results")
    
