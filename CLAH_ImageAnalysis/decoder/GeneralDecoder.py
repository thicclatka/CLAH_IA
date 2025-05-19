from typing import Dict, Literal, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm import tqdm

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical

    TF_IMPORTED = True
except ImportError:
    TF_IMPORTED = False
    print("\nWarning: TensorFlow not found. LSTM functionality will be unavailable.\n")


class GeneralDecoder:
    def __init__(self) -> None:
        pass

    @staticmethod
    def calc_CostParam(
        data_arr: np.ndarray,
        label_arr: np.ndarray,
        num_folds: int,
        kernel_type: str,
        gamma: str | float,
        weight: str,
        num_Cs: int = 30,
    ) -> float:
        """
        Calculates the best value of C for a given set of data and labels using cross-validation.

        Args:
            data_arr (numpy.ndarray): The input data array.
            label_arr (numpy.ndarray): The label array.
            num_folds (int): The number of folds for cross-validation.
            num_Cs (int, optional): The number of candidate C values to evaluate. Defaults to 30.

        Returns:
            float: The best value of C.

        """
        best_C = 0
        best_accuracy = 0

        candidate_C_values = np.logspace(-2, 2, num_Cs)

        with tqdm(candidate_C_values, desc="Evaluating C") as t:
            for C in t:
                t.set_description(f"Evaluating C: {C:.4f}")
                fold_accuracies = []
                fold_accuracies, _, _ = GeneralDecoder.run_Decoder(
                    data_arr=data_arr,
                    label_arr=label_arr,
                    num_folds=num_folds,
                    decoder_type="SVC",
                    C=C,
                    kernel_type=kernel_type,
                    gamma=gamma,
                    weight=weight,
                )
                accu2check = np.max(fold_accuracies)
                if accu2check >= best_accuracy:
                    best_accuracy = accu2check
                    best_C = C
                # Update the postfix to show the current best C and its accuracy
                t.set_postfix(
                    best_C=f"{best_C:.4f}",
                    best_acc=f"{best_accuracy:.4f}",
                    last_acc=f"{accu2check:.4f}",
                )

        return best_C

    @staticmethod
    def run_Decoder(
        data_arr: np.ndarray,
        label_arr: np.ndarray,
        num_folds: int,
        decoder_type: Literal["SVC", "GBM", "NB", "KNN"],
        random_state: int = 14,
        **kwargs,
    ) -> tuple:
        """
        Run the decoder on the given data.

        Parameters:
        - data_arr (ndarray): The input data array with shape (X, features). X can be n_cells, trials, cue epochs, etc.
        - label_arr (ndarray): The label array with shape (X,).
        - num_folds (int): The number of folds for cross-validation.
        - C (float): The cost parameter for the SVM model.

        Returns:
        - accuracy (ndarray): The accuracy scores for each timepoint and fold.
        - conf_matrices (ndarray): The confusion matrices for each timepoint and fold.
        """
        try:
            if decoder_type == "SVC":
                C = kwargs.get("C", 1.0)
                kernel_type = kwargs.get("kernel_type", "rbf")
                gamma = kwargs.get("gamma", "scale")
                weight = kwargs.get("weight", None)
                verbose = kwargs.get("verbose", False)
            elif decoder_type == "GBM":
                n_estimators = kwargs.get("n_estimators", 100)
                max_depth = kwargs.get("max_depth", 3)
                learning_rate = kwargs.get("learning_rate", 0.1)
            elif decoder_type == "NB":
                pass
            elif decoder_type == "KNN":
                n_neighbors = kwargs.get("n_neighbors", 5)
                weights = kwargs.get("weights", "uniform")
                metric = kwargs.get("metric", "cosine")
        except Exception as e:
            raise ValueError(
                f"Error processing decoder parameters for {decoder_type}: {e}. Check kwargs: {kwargs}"
            )

        if decoder_type not in ["SVC", "GBM", "NB", "KNN"]:
            raise ValueError(
                "Unsupported decoder_type. Must be 'SVC', 'GBM', 'NB', or 'KNN'."
            )

        cv = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=random_state
        )

        fold_accuracy = []
        fold_conf_matrices = []
        fold_medae = []  # List to store MedAE for each fold

        for train, test in cv.split(data_arr, label_arr):
            XTrain = data_arr[train, :]
            YTrain = label_arr[train]

            XTest = data_arr[test, :]
            YTest = label_arr[test]

            if decoder_type == "SVC":
                svmmodel = SVC(
                    C=C,
                    kernel=kernel_type,
                    gamma=gamma,
                    verbose=verbose,
                    class_weight=weight,
                    random_state=random_state,
                )
                svmmodel.fit(XTrain, YTrain)

                predictions = svmmodel.predict(XTest)
            elif decoder_type == "GBM":
                gbmmodel = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state,
                )
                gbmmodel.fit(XTrain, YTrain)

                predictions = gbmmodel.predict(XTest)
            elif decoder_type == "NB":
                nbmodel = GaussianNB()
                nbmodel.fit(XTrain, YTrain)

                predictions = nbmodel.predict(XTest)
            elif decoder_type == "KNN":
                knn_model = KNeighborsClassifier(
                    n_neighbors=n_neighbors, weights=weights, metric=metric
                )
                knn_model.fit(XTrain, YTrain)
                predictions = knn_model.predict(XTest)

            # get accuracy for the current fold
            acc = accuracy_score(YTest, predictions)
            fold_accuracy.append(acc)

            # confusion matrix
            cm = confusion_matrix(YTest, predictions)
            fold_conf_matrices.append(cm)

            medae = np.median(np.abs(YTest - predictions))
            fold_medae.append(medae)

        # Return numpy arrays for all metrics
        return (
            np.array(fold_accuracy),
            np.array(fold_conf_matrices),
            np.array(fold_medae),
        )

    # @staticmethod
    # def decode_via_LSTM(
    #     data_arr: np.ndarray,
    #     label_arr: np.ndarray,
    #     num_folds: int,
    #     random_state: int = 14,
    # ) -> tuple:
    #     input_shape = (data_arr.shape[1], data_arr.shape[2])  # timepoints x neurons
    #     num_classes = int(len(np.unique(label_arr)))

    #     cat_arr = to_categorical(label_arr - 1, num_classes=num_classes)

    #     accuracy = []
    #     confusion_matrices = []
    #     cv = StratifiedKFold(
    #         n_splits=num_folds, shuffle=True, random_state=random_state
    #     )
    #     for idx, (train, test) in enumerate(
    #         cv.split(np.zeros(len(label_arr)), label_arr)
    #     ):
    #         print(f"Fold: {idx + 1}")
    #         XTrain = data_arr[train, :, :]
    #         YTrain = cat_arr[train]

    #         XTest = data_arr[test, :, :]
    #         YTest = cat_arr[test]

    #         model = GeneralDecoder._build_LSTM(input_shape, num_classes)
    #         model.fit(
    #             XTrain,
    #             YTrain,
    #             epochs=50,
    #             batch_size=32,
    #             validation_split=0.2,
    #             verbose=0,
    #         )

    #         predictions = model.predict(XTest)
    #         YPred = np.argmax(predictions, axis=1)
    #         YTrue = np.argmax(YTest, axis=1)

    #         accu = accuracy_score(YTrue, YPred)
    #         accuracy.append(accu)

    #         cm = confusion_matrix(YTrue, YPred)
    #         confusion_matrices.append(cm)

    #     return np.array(accuracy), np.array(confusion_matrices)

    # @staticmethod
    # def _build_LSTM(
    #     input_shape: tuple,
    #     num_classes: int,
    #     return_sequences: bool = True,
    #     first_layer_units: int = 128,
    #     second_layer_units: int = 64,
    #     dropout: float = 0.5,
    #     activation: str = "softmax",
    #     optimizer: str = "adam",
    #     loss="categorical_crossentropy",
    #     metrics=["accuracy"],
    # ):
    #     """
    #     Build LSTM model.
    #     """
    #     model = Sequential()
    #     model.add(
    #         LSTM(
    #             units=first_layer_units,
    #             input_shape=input_shape,
    #             return_sequences=return_sequences,
    #         )
    #     )
    #     model.add(Dropout(rate=dropout))
    #     model.add(LSTM(units=second_layer_units))
    #     model.add(Dropout(rate=dropout))
    #     model.add(Dense(units=num_classes, activation=activation))

    #     model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    #     return model

    @staticmethod
    def calculate_ConfusionMatrix_metrics(cm: np.ndarray) -> dict:
        """
        Calculate the confusion matrix metrics.

        Args:
            cm (numpy.ndarray): The confusion matrix.

        Returns:
            dict: The confusion matrix metrics.

        """
        if cm.shape[0] != cm.shape[1]:
            raise ValueError("Confusion matrix must be square!")

        # calculate the accuracy
        accuracy = np.trace(cm) / np.sum(cm)

        # calculate the precision
        precision = np.diag(cm) / np.sum(cm, 0)

        # calculate the recall
        recall = np.diag(cm) / np.sum(cm, 1)

        # calculate the F1 score
        f1 = 2 * precision * recall / (precision + recall)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def _build_LSTM_prediction(
        input_shape: tuple, output_units: int, **kwargs
    ) -> "tf.keras.models.Sequential":
        """
        Builds a simple LSTM model for sequence prediction (regression).

        Args:
            input_shape (tuple): Shape of the input sequences (seq_length, n_features).
            output_units (int): Number of output units (should match n_features for predicting next step).
            **kwargs: Optional hyperparameters for the model.
                lstm1_units (int): Units in the first LSTM layer (default: 64).
                lstm2_units (int): Units in the second LSTM layer (default: 32). If 0, layer is skipped.
                dropout_rate (float): Dropout rate (default: 0.3).
                optimizer (str): Keras optimizer name (default: 'adam').

        Returns:
            A compiled Keras Sequential model.

        Raises:
            ImportError: If TensorFlow/Keras is not installed.
        """
        if not TF_IMPORTED:
            raise ImportError(
                "TensorFlow/Keras is required for LSTM functionality but not found."
            )

        # Get hyperparameters from kwargs or use defaults
        lstm1_units = kwargs.get("lstm1_units", 64)
        lstm2_units = kwargs.get("lstm2_units", 32)  # Set to 0 to disable
        dropout_rate = kwargs.get("dropout_rate", 0.3)
        optimizer = kwargs.get("optimizer", "adam")

        model = Sequential(name="LSTM_Predictor")
        # Optional: Masking layer if using padding with a specific value (e.g., if 0 padding is bad)
        # model.add(Masking(mask_value=0., input_shape=input_shape)) # If using masking
        # model.add(LSTM(units=lstm1_units, input_shape=input_shape, return_sequences=lstm2_units > 0)) # Adjust input_shape if using Masking

        # Simpler start without masking
        model.add(
            LSTM(
                units=lstm1_units,
                input_shape=input_shape,
                return_sequences=lstm2_units > 0,
            )
        )
        model.add(Dropout(rate=dropout_rate))

        if lstm2_units > 0:
            model.add(
                LSTM(units=lstm2_units, return_sequences=False)
            )  # Last LSTM before Dense should not return sequences
            model.add(Dropout(rate=dropout_rate))

        # Output layer for regression: linear activation, number of units = number of features to predict
        model.add(Dense(units=output_units, activation="linear"))

        # Compile for regression
        model.compile(
            optimizer=optimizer, loss="mean_squared_error", metrics=["mae"]
        )  # Use MSE loss

        # print(model.summary())  # Optional: Print model summary
        return model

    @staticmethod
    def decode_via_LSTM_prediction(
        X_lstm: np.ndarray,
        y_lstm: np.ndarray,
        sequence_labels: np.ndarray,
        num_folds: int,
        random_state: int = 14,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decodes sequences using LSTM for predicting the next time step across all features.
        Performs cross-validation and returns performance metrics for each fold.

        Args:
            X_lstm: Input sequences, shape (n_sequences, seq_length, n_features).
            y_lstm: Target values (next time step), shape (n_sequences, n_features).
            sequence_labels: Labels indicating the origin (e.g., lap type) of each sequence, used for stratified splitting. Shape (n_sequences,).
            num_folds: Number of folds for cross-validation.
            random_state: Random state for reproducibility.
            **kwargs: Optional keyword arguments passed to _build_LSTM_prediction and model.fit.Includes epochs (default: 50), batch_size (default: 32), verbose (default: 0), lstm1_units, lstm2_units, dropout_rate, optimizer, use_early_stopping (bool, default: False), patience (int, default: 5). # Add other relevant fit params

        Returns:
            A tuple containing:
            - mse_scores (np.ndarray): Mean Squared Error on the test set for each fold.
            - mae_scores (np.ndarray): Mean Absolute Error on the test set for each fold.
            - r2_scores (np.ndarray): R-squared score on the test set for each fold.

        Raises:
            ImportError: If TensorFlow/Keras is not installed.
            ValueError: If input array dimensions are incorrect.
        """
        if not TF_IMPORTED:
            raise ImportError(
                "TensorFlow/Keras is required for LSTM functionality but not found."
            )

        if X_lstm.ndim != 3 or y_lstm.ndim != 2:
            raise ValueError(
                f"Incorrect input dimensions. X_lstm should be 3D (sequences, timesteps, features), y_lstm should be 2D (sequences, features). Got X: {X_lstm.shape}, y: {y_lstm.shape}"
            )
        if X_lstm.shape[0] != y_lstm.shape[0]:
            raise ValueError(
                f"Number of sequences in X_lstm ({X_lstm.shape[0]}) and y_lstm ({y_lstm.shape[0]}) must match."
            )
        if X_lstm.shape[2] != y_lstm.shape[1]:
            raise ValueError(
                f"Number of features in X_lstm ({X_lstm.shape[2]}) and y_lstm ({y_lstm.shape[1]}) must match."
            )
        if X_lstm.shape[0] != len(sequence_labels):
            raise ValueError(
                f"Number of sequences in X_lstm ({X_lstm.shape[0]}) and sequence_labels ({len(sequence_labels)}) must match."
            )

        n_sequences = X_lstm.shape[0]
        seq_length = X_lstm.shape[1]
        n_features = X_lstm.shape[2]  # Also the number of output units
        input_shape = (seq_length, n_features)

        mse_scores = []
        mae_scores = []
        r2_scores = []

        # Use StratifiedKFold if labels are provided and have more than one class
        if sequence_labels is not None and len(np.unique(sequence_labels)) > 1:
            print(
                f"Using StratifiedKFold based on {len(np.unique(sequence_labels))} unique sequence labels."
            )
            cv = StratifiedKFold(
                n_splits=num_folds, shuffle=True, random_state=random_state
            )
            split_iterator = cv.split(
                np.zeros(n_sequences), sequence_labels
            )  # Use labels for stratification
        else:
            if sequence_labels is None:
                print("No sequence labels provided, using standard KFold.")
            else:
                print(
                    f"Only {len(np.unique(sequence_labels))} unique sequence label found, using standard KFold."
                )
            cv = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
            split_iterator = cv.split(np.zeros(n_sequences))  # Just split indices

        # Training arguments from kwargs
        epochs = kwargs.get("epochs", 50)
        batch_size = kwargs.get("batch_size", 32)
        verbose_fit = kwargs.get("verbose", 0)
        use_early_stopping = kwargs.get("use_early_stopping", False)
        patience = kwargs.get("patience", 5)

        fold_num = 0
        for train_idx, test_idx in tqdm(
            split_iterator, total=num_folds, desc="LSTM Folds"
        ):
            fold_num += 1
            XTrain, XTest = X_lstm[train_idx], X_lstm[test_idx]
            YTrain, YTest = y_lstm[train_idx], y_lstm[test_idx]

            model = GeneralDecoder._build_LSTM_prediction(
                input_shape, n_features, **kwargs
            )

            callbacks = []
            if use_early_stopping:
                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    verbose=verbose_fit,
                    restore_best_weights=True,
                )
                callbacks.append(early_stop)

            history = model.fit(
                XTrain,
                YTrain,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1 if use_early_stopping else None,
                callbacks=callbacks,
                verbose=verbose_fit,
            )

            loss, mae = model.evaluate(XTest, YTest, verbose=0)
            mse_scores.append(loss)
            mae_scores.append(mae)

            y_pred = model.predict(XTest, verbose=0)
            r2 = r2_score(YTest, y_pred, multioutput="variance_weighted")
            r2_scores.append(r2)

            # print(f"Fold {fold_num} Test MSE: {loss:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        print("LSTM Cross-validation finished.")
        print(f"MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
        print(f"MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
        print(f"R2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        return np.array(mse_scores), np.array(mae_scores), np.array(r2_scores)
