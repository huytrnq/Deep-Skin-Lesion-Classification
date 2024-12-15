from pathlib import Path
from tqdm import tqdm
import numpy as np

import dagshub
import mlflow.pytorch
import torch
from torch.nn.functional import softmax
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    classification_report,
)

from utils.utils import load_data_file, load_config, build_transforms
from utils.dataset import SkinDataset


class DempsterShaferEnsemble:
    def __init__(
        self,
        run_ids,
        model_artifact="skin_lesion_model",
        config_artifact="config/config.json",
        device="cpu",
    ):
        """
        Initialize the ensemble.

        Args:
            run_ids (list): List of MLflow run IDs for the models.
            model_artifact (str): Path to the model artifact in the MLflow run.
            config_artifact (str): Path to the configuration file artifact in the MLflow run.
            device (str): Device to run the models (e.g., "cuda" or "cpu").
        """
        self.run_ids = run_ids
        self.model_artifact = model_artifact
        self.config_artifact = config_artifact
        self.device = device
        self.loaded_model = None
        self.current_run_id = None
        self.configs = {}

    def load_mlflow_run(self, run_id):
        """
        Load MLflow run for the given run ID.

        Args:
            run_id (str): MLflow run ID.

        Returns:
            mlflow.entities.Run: MLflow run object.
        """
        self.current_run_id = run_id
        self.loaded_model = mlflow.pytorch.load_model(
            f"runs:/{run_id}/{self.model_artifact}"
        ).to(self.device)
        self.loaded_model.eval()
        self.load_config(run_id)

    def load_model(self, run_id):
        """
        Dynamically load a model from MLflow.

        Args:
            run_id (str): MLflow run ID for the model.

        Returns:
            torch.nn.Module: Loaded model.
        """
        if self.current_run_id != run_id:
            if self.loaded_model is not None:
                # Move the current model to CPU and delete it
                self.loaded_model.to("cpu")
                del self.loaded_model
                torch.cuda.empty_cache()  # Free GPU memory

            print(f"Loading model for run_id: {run_id}")
            model_uri = f"runs:/{run_id}/{self.model_artifact}"
            self.loaded_model = mlflow.pytorch.load_model(model_uri).to(self.device)
            self.loaded_model.eval()
            self.current_run_id = run_id
        return self.loaded_model

    def load_config(self, run_id):
        """
        Dynamically load configuration from MLflow.

        Args:
            run_id (str): MLflow run ID.

        Returns:
            dict: Configuration dictionary.
        """
        if run_id not in self.configs:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=self.config_artifact
            )
            if Path(local_path).is_file():
                print(f"Config file downloaded to: {local_path}")
                self.configs[run_id] = load_config(local_path)
            else:
                raise FileNotFoundError(f"Config file not found for run_id: {run_id}")
        return self.configs[run_id]

    def predict(self, dataloader, tta=False, num_tta=5):
        """
        Perform ensemble prediction using Dempster-Shafer Theory.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for test data.
            tta (bool): Whether to use Test Time Augmentation (TTA).
            num_tta (int): Number of TTA iterations if TTA is enabled.

        Returns:
            tuple: Final predictions (list) and true labels (np.ndarray).
        """
        combined_evidence_list = []  # Evidence for all samples across all runs
        all_labels = []  # Ground truth labels

        for run_id in self.run_ids:
            print(f"Processing run_id: {run_id}")
            model = self.load_model(run_id)
            config = self.load_config(run_id)
            train_transform = build_transforms(config["transformations"]["train"])
            test_transform = build_transforms(config["transformations"]["test"])

            for images, batch_labels in tqdm(
                dataloader, desc=f"Predicting for run_id: {run_id}"
            ):
                # Store ground truth labels (only once)
                all_labels.extend(batch_labels.numpy())

                # Apply TTA or standard transformation
                if tta:
                    # Perform TTA
                    tta_probs = []
                    for _ in range(num_tta):
                        augmented_images = [
                            train_transform(transforms.ToPILImage()(img.detach().cpu()))
                            for img in images
                        ]
                        augmented_images = torch.stack(augmented_images).to(self.device)
                        output = model(augmented_images)
                        tta_probs.append(softmax(output, dim=1))
                    avg_probs = torch.stack(tta_probs).mean(dim=0)
                else:
                    # Standard inference
                    transformed_images = torch.stack(
                        [
                            test_transform(transforms.ToPILImage()(img.detach().cpu()))
                            for img in images
                        ]
                    ).to(self.device)
                    output = model(transformed_images)
                    avg_probs = softmax(output, dim=1)

                # Convert probabilities to evidence
                for prob in avg_probs:
                    evidence = self.get_evidence_from_probabilities(prob.cpu().numpy())
                    combined_evidence_list.append(evidence)

        # Combine evidence across runs using Dempster-Shafer Theory
        n_samples = len(all_labels) // len(self.run_ids)  # Number of unique samples
        evidence_per_sample = len(self.run_ids)  # Evidence contributions per sample

        final_predictions = []
        for i in range(n_samples):
            sample_evidence_list = combined_evidence_list[
                i * evidence_per_sample : (i + 1) * evidence_per_sample
            ]
            combined_evidence = self.dempster_shafer_combination(sample_evidence_list)
            predicted_class = max(
                combined_evidence, key=combined_evidence.get
            )  # Class with highest belief
            final_predictions.append(predicted_class)

        return final_predictions, np.array(all_labels[:n_samples])

    @staticmethod
    def get_evidence_from_probabilities(probs):
        """
        Convert probabilities to Dempster-Shafer evidence.

        Args:
            probs (list or torch.Tensor): Predicted probabilities for each class.

        Returns:
            dict: Evidence for each hypothesis and uncertainty.
        """
        probs = probs.tolist()
        evidence = {f"Class_{i}": prob for i, prob in enumerate(probs)}
        evidence["uncertainty"] = 1 - sum(probs)
        return evidence

    @staticmethod
    def dempster_shafer_combination(evidence_list):
        """
        Combine multiple pieces of evidence using Dempster-Shafer combination rule.

        Args:
            evidence_list (list): List of evidence dictionaries.

        Returns:
            dict: Combined evidence.
        """
        combined_evidence = evidence_list[0]
        for evidence in evidence_list[1:]:
            new_combined_evidence = {}
            for hypo_i, belief_i in combined_evidence.items():
                for hypo_j, belief_j in evidence.items():
                    if hypo_i == hypo_j:
                        new_combined_evidence[hypo_i] = (
                            new_combined_evidence.get(hypo_i, 0) + belief_i * belief_j
                        )
                    elif hypo_i != "uncertainty" and hypo_j != "uncertainty":
                        new_combined_evidence["conflict"] = (
                            new_combined_evidence.get("conflict", 0)
                            + belief_i * belief_j
                        )

            # Normalize to resolve conflict
            conflict = new_combined_evidence.get("conflict", 0)
            if conflict < 1:
                for hypo in combined_evidence:
                    if hypo != "conflict":
                        new_combined_evidence[hypo] = new_combined_evidence.get(
                            hypo, 0
                        ) / (1 - conflict)
                new_combined_evidence.pop("conflict", None)

            combined_evidence = new_combined_evidence
        return combined_evidence


class Ensemble:
    def __init__(self, run_ids, mode="majority", tta=False):
        """
        Initialize the ensemble.

        Args:
            run_ids (list): List of MLflow run IDs for the models.
            mode (str): Ensemble mode ("majority" or "average").
            tta (bool): Whether to use Test Time Augmentation (TTA).
        """
        self.run_ids = run_ids
        self.mode = mode
        self.tta = tta
        self.artifact_path = (
            "results/prediction_probs.npy"
            if not tta
            else "results/tta_prediction_probs.npy"
        )
        self.predictions = []
        dagshub.init(
            repo_owner="huytrnq",
            repo_name="Deep-Skin-Lesion-Classification",
            mlflow=True,
        )

    def load_predictions_from_mlflow(self):
        """
        Load predictions from MLflow artifacts.
        """
        for run_id in self.run_ids:
            with mlflow.start_run(run_id=run_id):
                predictions_npy_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=self.artifact_path
                )
                self.predictions.append(np.load(predictions_npy_path))

    def majority_voting(self):
        """
        Perform majority voting on the predictions.

        Returns:
            list: Final predictions.
        """
        final_predictions = []
        for i in range(len(self.predictions[0])):
            votes = [np.argmax(pred[i]) for pred in self.predictions]
            final_predictions.append(max(set(votes), key=votes.count))
        return final_predictions

    def average_voting(self):
        """
        Perform average voting on the predictions.

        Returns:
            list: Final predictions after averaging probabilities.
        """
        final_predictions = []
        for i in range(len(self.predictions[0])):
            # Collect the probabilities for the i-th sample across all models
            votes = [pred[i] for pred in self.predictions]

            # Compute the average probabilities
            averaged_probs = np.mean(votes, axis=0)

            # Select the class with the highest average probability
            final_predictions.append(np.argmax(averaged_probs))
        return final_predictions

    def predict(self):
        """
        Perform majority voting on the predictions and return the final predictions.
        """
        self.load_predictions_from_mlflow()
        if self.mode == "average":
            final_predictions = self.average_voting()
        else:
            final_predictions = self.majority_voting()
        return final_predictions


if __name__ == "__main__":
    dagshub.init(
        repo_owner="huytrnq", repo_name="Deep-Skin-Lesion-Classification", mlflow=True
    )
    run_ids = [
        "d14d502cca984bcb8b0e9f66deec8cd2",
        "73f0abbe48dc4ca19cdb9b74a1826521",
    ]  # Replace with actual run IDs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the ensemble
    ensemble = DempsterShaferEnsemble(run_ids, device=DEVICE)

    # Load test data
    test_names, test_labels = load_data_file("datasets/Binary/val.txt")
    test_transform = transforms.Compose(
        [transforms.Resize((640, 640)), transforms.ToTensor()]
    )  # Basic transform
    test_dataset = SkinDataset(
        root_path="/root/huy/datasets/Binary",
        sub_folder="val",
        names=test_names,
        labels=test_labels,
        transform=test_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Perform predictions
    with torch.no_grad():
        final_predictions, labels = ensemble.predict(test_loader, tta=False, num_tta=3)
        print("There are {} conflicts".format(final_predictions.count("conflict")))
        final_predictions = [
            float(pred.split("_")[-1]) if "conflict" not in pred else 0
            for pred in final_predictions
        ]

    # Calculate metrics
    kappa = cohen_kappa_score(labels, final_predictions)
    accuracy = accuracy_score(labels, final_predictions)
    f1 = f1_score(labels, final_predictions)
    report = classification_report(labels, final_predictions)
    print(f"Kappa: {kappa:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(report)

    # print("Ensemble Predictions with TTA:", prediction_probs)
