import os
import glob
import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import sqlite3
from datetime import datetime
from tqdm import tqdm
from scipy.signal import welch
from skimage.measure import regionprops
from skimage.morphology import label
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.signal import find_peaks
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from CLAH_ImageAnalysis.utils import paths
from CLAH_ImageAnalysis.utils import db_utils
from CLAH_ImageAnalysis.utils import text_dict
import streamlit as st


class NNModel4BinaryClassification(nn.Module):
    def __init__(
        self,
        feature_size: int,
    ) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.Linear(feature_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class GeneralModelRunner:
    def __init__(
        self,
        model: nn.Module,
        features_order: list[str],
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_splits: int = 10,
        stratify: bool = True,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.features_order = features_order
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.stratify = stratify
        self.random_state = random_state

    def train_model(
        self,
        features_vector: torch.Tensor,
        labels_vector: torch.Tensor,
        n_epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        patience: int = 10,
        warmup_epochs: int = 5,
        warmup_factor: float = 0.1,
    ) -> None:
        """
        Train the model

        Args:
            features_vector (torch.Tensor): Features vector
            labels_vector (torch.Tensor): Labels vector
            n_epochs (int | None, optional): Number of epochs to train. Defaults to None, which uses the values set in __init__.
            learning_rate (float | None, optional): Learning rate. Defaults to None, which uses the values set in __init__.
            batch_size (int | None, optional): Batch size. Defaults to None, which uses the values set in __init__.
            patience (int, optional): Patience for early stopping. Number of epochs to wait before stopping if no improvement is seen. Defaults to 10.
            warmup_epochs (int, optional): After which epoch to start adjusting learning rate. Defaults to 5.
            warmup_factor (float, optional): Factor to adjust learning rate by during warmup. Defaults to 0.1.
        """
        # set parameters if not provided
        n_epoch2use = n_epochs if n_epochs is not None else self.n_epochs
        learning_rate2use = (
            learning_rate if learning_rate is not None else self.learning_rate
        )
        batch_size2use = batch_size if batch_size is not None else self.batch_size

        num_accepted = (labels_vector == 1).sum()
        num_rejected = (labels_vector == 0).sum()
        total_components = len(labels_vector)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate2use, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Calculate positive weight for imbalanced dataset
        pos_weight = torch.tensor(
            [(num_rejected / num_accepted) if num_accepted > 0 else 1]
        )
        class_weights = torch.tensor(
            [
                total_components / (2 * num_rejected),
                total_components / (2 * num_accepted),
            ]
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        weights = [class_weights[int(label)] for label in labels_vector]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

        # setup parameters necessary for early stopping
        patience_counter = 0
        best_loss = float("inf")

        pbar = tqdm(range(n_epoch2use), desc=f"Training model")
        for epoch in pbar:
            pbar.set_description(f"Training model: Epoch {epoch + 1}")
            # Adjust warmup rate after first 5 epochs
            if epoch < warmup_epochs:
                # Gradually increase learning rate
                lr_scale = warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs)
                current_lr = learning_rate2use * lr_scale
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            self.model.train()  # Set to training mode
            perm = torch.randperm(features_vector.size(0))
            running_loss = 0.0

            indices = list(sampler)
            for i in range(0, len(indices), batch_size2use):
                indices = indices[i : i + batch_size2use]
                batch_x = features_vector[indices]
                batch_y = labels_vector[indices]

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / (features_vector.size(0) / batch_size2use)
            scheduler.step(epoch_loss)

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}\n")
                break

            if epoch % 10 == 0:
                tqdm.write(
                    f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

    def cross_validate(
        self,
        features_vector: torch.Tensor,
        labels_vector: torch.Tensor,
        metadata: dict,
        n_splits: int | None = None,
        stratify: bool | None = None,
        random_state: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        n_epochs: int | None = None,
    ):
        # initialize metrics
        self.metrics = {}

        best_f1 = 0
        best_state_dict = None
        best_state_dict_fold_idx = None

        # set parameters
        if stratify is None:
            stratify = self.stratify
        if n_splits is None:
            n_splits = self.n_splits
        if random_state is None:
            random_state = self.random_state
        if batch_size is None:
            batch_size = self.batch_size
        if learning_rate is None:
            learning_rate = self.learning_rate
        if n_epochs is None:
            n_epochs = self.n_epochs

        if stratify:
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
        }

        pbar = tqdm(
            cv.split(features_vector, labels_vector),
            total=n_splits,
            desc=f"Cross-validating",
        )
        for fold, (train_idx, val_idx) in enumerate(pbar):
            pbar.set_description(
                f"Cross-validating: Fold {fold + 1} (Best F1: {best_f1:.4f} | Best fold: {best_state_dict_fold_idx + 1 if best_state_dict_fold_idx is not None else 'N/A'})"
            )
            XTrain = features_vector[train_idx]
            YTrain = labels_vector[train_idx]
            XTest = features_vector[val_idx]
            YTest = labels_vector[val_idx]

            # reset weights before each fold training
            self.model.apply(self._weight_reset)

            self.train_model(
                XTrain,
                YTrain,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )

            val_preds = self.predict_binary(XTest)
            val_probs = self.get_component_scores(XTest)

            fold_metrics = self._calculate_matrics(YTest, val_preds, val_probs)
            for metric in metrics.keys():
                metrics[metric].append(fold_metrics[metric])

            # Save the model state if it has the highest F1 score so far
            if fold_metrics["f1"] > best_f1:
                best_f1 = fold_metrics["f1"]
                best_state_dict_fold_idx = fold
                best_state_dict = self.model.state_dict().copy()

        # Restore the best model state
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        metrics_mean = {
            metric: (np.mean(metrics[metric]), np.std(metrics[metric]))
            for metric in metrics.keys()
        }

        # save metrics into self
        self.metrics = {
            "per_fold": metrics,
            "mean": metrics_mean,
            "metadata": metadata.copy(),
            "best_fold_idx": best_state_dict_fold_idx,
        }

    def _calculate_matrics(
        self, YTest: torch.Tensor, val_preds: torch.Tensor, val_probs: np.ndarray
    ) -> dict:
        """Calculate metrics for the validation set"""
        YTest = YTest.numpy().flatten()
        val_preds = val_preds.numpy().flatten()
        val_probs = val_probs.flatten()

        return {
            "accuracy": accuracy_score(YTest, val_preds),
            "precision": precision_score(YTest, val_preds),
            "recall": recall_score(YTest, val_preds),
            "f1": f1_score(YTest, val_preds),
            "auc": roc_auc_score(YTest, val_probs),
        }

    def evaluate_components(self, features_vector: torch.Tensor) -> torch.Tensor:
        """Evaluate components using the trained model"""
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            predictions = self.model(features_vector)
        return predictions

    def predict_binary(
        self, features_vector: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Get binary predictions (accepted/rejected) for components"""
        predictions = self.evaluate_components(features_vector)
        return (predictions > threshold).float()

    def get_component_scores(self, features_vector: torch.Tensor) -> np.ndarray:
        """Get probability scores for all components"""
        predictions = self.evaluate_components(features_vector)
        return predictions.numpy().flatten()

    @staticmethod
    def _weight_reset(m: nn.Module) -> None:
        """Reset the weights of the model"""
        if isinstance(m, nn.Linear):
            m.reset_parameters()

    def save_model(
        self,
        fname: str,
        path: str | None = None,
        model_type: str = "BINARY_CLASSIFIER",
    ) -> str:
        """
        Save trained model to file

        Parameters:
            fname (str): Name of the model
            path (str | None, optional): Path to save the model. Defaults to None.

        Returns:
            str: Path to saved model
        """

        PTH = text_dict()["file_tag"]["PYTORCH"]
        numSess = self.metrics["metadata"]["total_sessions"]
        curr_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        # add timestamp to fname
        fname = f"{fname}_numSess{numSess}_{model_type}_CV{self.n_splits}_{curr_date}"

        if not fname.endswith(PTH):
            fname = fname + PTH

        # by default save to NNmodels folder
        if path is None:
            path = paths.get_path2NNmodels()

        fnameWpath = str(Path(path, fname))

        input_size = None
        for layer in self.model.classifier:
            if isinstance(layer, nn.Linear):
                input_size = layer.in_features
                break

        if path.exists():
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "features_order": self.features_order,  # Save feature configuration
                    "model_config": {
                        "input_size": input_size,
                    },
                    "metrics": self.metrics,
                },
                fnameWpath,
            )
        else:
            raise ValueError(f"{path} does not exist")

        return str(fnameWpath)

    @classmethod
    def load_model(cls, path2pth: str) -> tuple["GeneralModelRunner", dict]:
        checkpoint = torch.load(path2pth)

        model = NNModel4BinaryClassification(
            feature_size=checkpoint["model_config"]["input_size"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Create runner with loaded model
        runner = cls(model, features_order=checkpoint["features_order"])
        runner.metrics = checkpoint["metrics"]

        # Return metadata if it exists
        metadata = checkpoint.get("metadata", None)

        return runner, metadata

    def predict_new_data(
        self,
        CTemp: np.ndarray,
        ASpat: np.ndarray,
        dim4spatial: tuple,
        freq_sampling: float = 30,
        freq_threshold: float = 1.0,
    ) -> dict:
        """Predict new data using the trained model"""
        FE = FeatureExtractor4segDict(
            CTempList=[CTemp],
            ASpatList=[ASpat],
            dim4spatialList=[dim4spatial],
            freq_sampling=freq_sampling,
            freq_threshold=freq_threshold,
        )
        features_vector = FE.create_Vectors(train=False)

        self.model.eval()
        with torch.no_grad():
            scores = self.get_component_scores(features_vector)
            predictions = self.predict_binary(features_vector)

        return {
            "scores": scores,
            "predictions": predictions,
            "features_vector": features_vector,
        }

    def find_optimal_threshold(
        self,
        predictions: np.ndarray,
        labels2compare2: np.ndarray,
        thresholds: np.ndarray = np.arange(0.3, 0.7, 0.05),
    ) -> float:
        """Find optimal threshold that balances true positive and true negative rates with a given predictions and labels"""
        best_threshold = 0.5
        best_balance = float("-inf")

        for threshold in thresholds:
            binary_preds = (predictions > threshold).astype(int)

            # Calculate true positive and true negative rates
            tp = np.sum((binary_preds == 1) & (labels2compare2 == 1))
            tn = np.sum((binary_preds == 0) & (labels2compare2 == 0))
            total_pos = np.sum(labels2compare2 == 1)
            total_neg = np.sum(labels2compare2 == 0)

            tpr = tp / total_pos if total_pos > 0 else 0
            tnr = tn / total_neg if total_neg > 0 else 0

            # Balance metric (geometric mean of TPR and TNR)
            balance = np.sqrt(tpr * tnr)

            if balance > best_balance:
                best_balance = balance
                best_threshold = threshold

        return best_threshold


class FeatureExtractor4segDict:
    def __init__(
        self,
        CTempList: list[np.ndarray],
        ASpatList: list[np.ndarray],
        dim4spatialList: list[tuple],
        listExpNames: list[str] | None = None,
        labelList: list[np.ndarray] | None = None,
        freq_sampling: float = 30,
        freq_threshold: float = 1.0,
        baseline_percentile: float = 10,
        pks_height: float | None = None,
        pks_distance: int = 1,
        pks_prominence: float = 0.35,
    ) -> None:
        self.CTempList = CTempList
        self.ASpatList = ASpatList
        self.dim4spatialList = dim4spatialList
        self.listExpNames = listExpNames.copy() if listExpNames is not None else None
        self.labelList = labelList
        self.freq_sampling = freq_sampling
        self.freq_threshold = freq_threshold
        self.baseline_percentile = baseline_percentile
        self.pks_height = pks_height
        self.pks_distance = pks_distance
        self.pks_prominence = pks_prominence
        self.features = {}

        self.cellTotals = [c.shape[0] for c in self.CTempList]

        self.allFeaturesDict = []
        self.allFeaturesVector = []

        self.features_order = [
            "mean",
            "std",
            "median",
            "q1",
            "q3",
            "iqr",
            "skewness",
            "kurtosis",
            "snr",
            "pnr",
            "num_peaks",
            "peak_frequency",
            "peak_heights_mean",
            "peak_heights_std",
            "peak_interval_mean",
            "peak_interval_std",
            "peak_amplitude_ratio",
            "temporal_correlation",
            "activity_ratio",
            "baseline",
            "dominant_frequency",
            "dynamic_range",
            "perimeter",
            "eccentricity",
            "solidity",
            "extent",
            "compactness",
            "max_intensity",
            "mean_intensity",
            "min_intensity",
            "centroid_r",
            "centroid_c",
            "sparsity",
            "nonzero_ratio",
        ]

    @staticmethod
    def _normalize_features(features: list) -> list[np.ndarray]:
        for sess_idx in range(len(features)):
            curr_sess_features = np.array(features[sess_idx])
            for i in range(curr_sess_features.shape[-1]):
                maxValues = np.max(curr_sess_features[:, i])
                minValues = np.min(curr_sess_features[:, i])
                curr_sess_features[:, i] = (curr_sess_features[:, i] - minValues) / (
                    maxValues - minValues + 1e-8
                )
            features[sess_idx] = curr_sess_features.astype(np.float32)
        return features

    def create_Vector4Features(self) -> torch.Tensor:
        featuresPerSession = []
        for sess_idx, cell_total in enumerate(self.cellTotals):
            featuresPerCell = []
            for cell_idx in tqdm(
                range(cell_total),
                desc=f"Extracting features from cells from S{sess_idx}",
            ):
                feature4cell = {}
                feature4cell["session_idx"] = sess_idx
                feature4cell["cell_idx"] = cell_idx

                temporal_trace = self.CTempList[sess_idx][cell_idx, :]
                spatial_trace = self.ASpatList[sess_idx][:, cell_idx]
                spatial_trace = spatial_trace.reshape(self.dim4spatialList[sess_idx])
                features4cell = self.extract_temporal_features(
                    featuresDict=feature4cell, temporal_trace=temporal_trace
                )
                features4cell = self.extract_spatial_features(
                    featuresDict=feature4cell, spatial_trace=spatial_trace
                )
                self.allFeaturesDict.append(features4cell)
                featuresPerCell.append(
                    self.prepare_features4model_perSession(feature4cell)
                )
            featuresPerSession.append(featuresPerCell)

        featuresPerSession = self._normalize_features(features=featuresPerSession)
        self.allFeaturesVector = torch.FloatTensor(
            np.concatenate(featuresPerSession, axis=0)
        )
        return self.allFeaturesVector

    def create_Vector4Labels(self) -> torch.Tensor:
        if self.labelList is None:
            raise ValueError("labelList is not provided")
        all_labels = np.concatenate(self.labelList)
        return torch.tensor(all_labels, dtype=torch.float32).reshape(-1, 1)

    def create_Vectors(
        self, train: bool = True
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Create vectors for training or testing

        Parameters:
            train (bool): Whether to create vectors for training or testing. If training, returns a tuple of features, labels, and metadata. If testing, returns only features.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | torch.Tensor: Vectors for training or testing
        """
        if train:
            features_vector = self.create_Vector4Features()
            labels_vector = self.create_Vector4Labels()
            metadata = self.export_features_metadata()
            return features_vector, labels_vector, metadata
        else:
            features_vector = self.create_Vector4Features()
            return features_vector

    def prepare_features4model_perSession(self, featuresDict: dict) -> list[float]:
        feature_vector = [featuresDict[feature] for feature in self.features_order]
        return feature_vector

    def extract_temporal_features(
        self, featuresDict: dict, temporal_trace: np.ndarray
    ) -> dict:
        # basic features
        featuresDict["mean"] = np.mean(temporal_trace)
        featuresDict["std"] = np.std(temporal_trace)
        featuresDict["median"] = np.median(temporal_trace)
        featuresDict["q1"] = np.percentile(temporal_trace, 25)
        featuresDict["q3"] = np.percentile(temporal_trace, 75)
        featuresDict["iqr"] = featuresDict["q3"] - featuresDict["q1"]
        featuresDict["skewness"] = skew(temporal_trace)
        featuresDict["kurtosis"] = kurtosis(temporal_trace)

        # normalize trace for pk detection
        norm_trace = (temporal_trace - temporal_trace.min()) / (
            temporal_trace.max() - temporal_trace.min()
        )
        if self.pks_height is None:
            pks_height2use = np.mean(norm_trace) + (2 * np.std(norm_trace))
        else:
            pks_height2use = self.pks_height

        peaks, _ = find_peaks(
            norm_trace,
            height=pks_height2use,
            distance=self.pks_distance,
            prominence=self.pks_prominence,
        )
        peak_heights = temporal_trace[peaks]
        if len(peaks) > 1:
            num_peaks = len(peaks)
            featuresDict["num_peaks"] = num_peaks
            featuresDict["peak_frequency"] = num_peaks / len(temporal_trace)
            featuresDict["peak_heights_mean"] = np.mean(peak_heights)
            featuresDict["peak_heights_std"] = np.std(peak_heights)
            featuresDict["peak_interval_mean"] = np.mean(np.diff(peaks))
            featuresDict["peak_interval_std"] = np.std(np.diff(peaks))
            featuresDict["peak_amplitude_ratio"] = np.max(peak_heights) / np.min(
                peak_heights
            )
        else:
            featuresDict["num_peaks"] = 0
            featuresDict["peak_frequency"] = 0
            featuresDict["peak_heights_mean"] = 0
            featuresDict["peak_heights_std"] = 0
            featuresDict["peak_interval_mean"] = 0
            featuresDict["peak_interval_std"] = 0
            featuresDict["peak_amplitude_ratio"] = 1

        featuresDict["baseline"] = np.percentile(
            temporal_trace, self.baseline_percentile
        )

        featuresDict["temporal_correlation"] = np.corrcoef(
            temporal_trace[:-1], temporal_trace[1:]
        )[0, 1]

        featuresDict["activity_ratio"] = np.sum(
            temporal_trace > featuresDict["baseline"]
        ) / len(temporal_trace)

        featuresDict["dynamic_range"] = (
            np.max(temporal_trace) - featuresDict["baseline"]
        )

        fft_trace = np.fft.fft(temporal_trace)
        power = np.abs(fft_trace) ** 2
        freq = np.fft.fftfreq(len(temporal_trace), 1 / self.freq_sampling)
        featuresDict["dominant_frequency"] = freq[np.argmax(power)]

        # SNR calculation
        freq, psd = welch(temporal_trace, fs=self.freq_sampling)
        signal_freq_mask = freq < self.freq_threshold
        noise_freq_mask = freq > self.freq_threshold

        signal_power = np.mean(psd[signal_freq_mask])
        noise_power = np.mean(psd[noise_freq_mask])
        featuresDict["snr"] = 10 * np.log10(signal_power / noise_power)

        # PNR calculation
        featuresDict["pnr"] = (
            np.max(temporal_trace) - np.mean(temporal_trace)
        ) / np.std(temporal_trace)
        return featuresDict

    def extract_spatial_features(
        self, featuresDict: dict, spatial_trace: np.ndarray
    ) -> dict:
        threshold = np.mean(spatial_trace) + (2 * np.std(spatial_trace))
        binary_mask = spatial_trace > threshold
        labeled_mask = label(binary_mask)

        if np.any(binary_mask):
            props = regionprops(labeled_mask, spatial_trace)[0]

            # Shape features
            featuresDict["area"] = props.area
            featuresDict["perimeter"] = props.perimeter
            featuresDict["eccentricity"] = props.eccentricity
            featuresDict["solidity"] = props.solidity
            featuresDict["extent"] = props.extent
            if props.perimeter > 0:
                featuresDict["compactness"] = (4 * np.pi * props.area) / (
                    props.perimeter * props.perimeter
                )
            else:
                featuresDict["compactness"] = 0

            # Intensity features
            featuresDict["max_intensity"] = props.max_intensity
            featuresDict["mean_intensity"] = props.mean_intensity
            featuresDict["min_intensity"] = props.min_intensity

            # Location features
            featuresDict["centroid_r"], featuresDict["centroid_c"] = props.centroid

            # Sparsity features
            featuresDict["sparsity"] = np.sum(binary_mask) / binary_mask.size
            featuresDict["nonzero_ratio"] = (
                np.count_nonzero(spatial_trace) / spatial_trace.size
            )
        else:
            # Default values if no signal is found
            default_value = 0
            shape_features = [
                "area",
                "perimeter",
                "eccentricity",
                "solidity",
                "extent",
                "compactness",
            ]
            intensity_features = ["max_intensity", "mean_intensity", "min_intensity"]
            location_features = ["centroid_r", "centroid_c"]
            sparsity_features = ["sparsity", "nonzero_ratio"]

            for feature in (
                shape_features
                + intensity_features
                + location_features
                + sparsity_features
            ):
                featuresDict[feature] = default_value

        return featuresDict

    def export_features_metadata(self) -> dict:
        total_sessions = len(self.CTempList)
        total_cells = sum(self.cellTotals)
        total_features = len(self.features_order)

        metadata = {
            "total_sessions": total_sessions,
            "total_cells": total_cells,
            "total_features": total_features,
            "features_used": self.features_order,
            "listExpNames": self.listExpNames,
        }
        return metadata


class dbHandler4CompEvaluator:
    def __init__(self, db_name: str) -> None:
        self.db_name = db_name
        self.db_path = db_utils.create_db_path_with_name(db_name)
        self.NNmodel_path = paths.get_path2NNmodels()

        self.refresh_db()

    def find_NNmodels4db(self):
        # create table from scratch, will overwrite any existing table

        PTH = text_dict()["file_tag"]["PYTORCH"]
        self.nnmodels = glob.glob(str(self.NNmodel_path / f"*{PTH}"))
        self.nnmodels.sort()

        for model_path in self.nnmodels:
            checkpoint = torch.load(model_path)
            all_metrics = checkpoint["metrics"]

            metadata = all_metrics["metadata"]
            mean_metrics = all_metrics["mean"]

            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO models (model_path, num_sessions, total_cells, total_features, listExpNames, features_used, accuracy, accuracy_std, precision, precision_std, recall, recall_std, f1, f1_std, auc, auc_std) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        model_path,
                        metadata["total_sessions"],
                        metadata["total_cells"],
                        metadata["total_features"],
                        json.dumps(metadata["listExpNames"]),
                        json.dumps(metadata["features_used"]),
                        mean_metrics["accuracy"][0],
                        mean_metrics["accuracy"][1],
                        mean_metrics["precision"][0],
                        mean_metrics["precision"][1],
                        mean_metrics["recall"][0],
                        mean_metrics["recall"][1],
                        mean_metrics["f1"][0],
                        mean_metrics["f1"][1],
                        mean_metrics["auc"][0],
                        mean_metrics["auc"][1],
                    ),
                )
                conn.commit()

    def create_models_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Drop the table if it exists
            c.execute("DROP TABLE IF EXISTS models")

            # create table from scratch
            c.execute("""CREATE TABLE IF NOT EXISTS models
                        (model_path text PRIMARY KEY,
                        num_sessions integer,
                        total_cells integer,
                        total_features integer,
                        listExpNames text,
                        features_used text,
                        accuracy real,
                        accuracy_std real,
                        precision real, 
                        precision_std real,
                        recall real,
                        recall_std real,
                        f1 real,
                        f1_std real,
                        auc real,
                        auc_std real)""")

            conn.commit()

    def refresh_db(self):
        self.create_models_db()
        self.find_NNmodels4db()

    def get_num_of_models(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM models")
            return c.fetchone()[0]

    def get_model_dict(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT model_path, num_sessions, total_cells, total_features, listExpNames, features_used, accuracy, accuracy_std, precision, precision_std, recall, recall_std, f1, f1_std, auc, auc_std FROM models"
            )

            models = c.fetchall()
            model_dict = {}
            for model in models:
                model_dict[model[0]] = {
                    "model_path": model[0],
                    "num_sessions": model[1],
                    "total_cells": model[2],
                    "total_features": model[3],
                    "listExpNames": json.loads(model[4]),
                    "features_used": json.loads(model[5]),
                    "accuracy": model[6],
                    "accuracy_std": model[7],
                    "precision": model[8],
                    "precision_std": model[9],
                    "recall": model[10],
                    "recall_std": model[11],
                    "f1": model[12],
                    "f1_std": model[13],
                    "auc": model[14],
                    "auc_std": model[15],
                }
            return model_dict

    def select_model_from_db(self, model_name: str) -> dict:
        model_dict = self.get_model_dict()
        return model_dict[model_name]
