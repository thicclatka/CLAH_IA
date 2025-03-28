import numpy as np
from typing import Literal
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


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
                fold_accuracies, _ = GeneralDecoder.run_Decoder(
                    data_arr, label_arr, num_folds, "SVC", C, kernel_type, gamma, weight
                )
                accu2check = np.max(np.mean(fold_accuracies, 1))
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
        decoder_type: Literal["SVC", "GBM", "LSTM", "NB"],
        random_state: int = 14,
        **kwargs,
    ) -> tuple:
        """
        Run the decoder on the given data.

        Parameters:
        - data_arr (ndarray): The input data array with shape (n_cells/pop, n_timepoints, n_trials).
        - label_arr (ndarray): The label array with shape (n_trials,).
        - num_folds (int): The number of folds for cross-validation.
        - C (float): The cost parameter for the SVM model.

        Returns:
        - accuracy (ndarray): The accuracy scores for each timepoint and fold.
        - conf_matrices (ndarray): The confusion matrices for each timepoint and fold.
        """
        if data_arr.ndim != 3:
            raise ValueError(
                "data_arr must be 3D array (n_cells/pop, n_timepoints, n_trials)!"
            )

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
        elif decoder_type == "LSTM":
            # transpose data to trials x timepoints x neurons
            data4LSTM = data_arr.transpose(2, 1, 0)
            accuracy, conf_matrices = GeneralDecoder.decode_via_LSTM(
                data_arr=data4LSTM,
                label_arr=label_arr,
                num_folds=num_folds,
                random_state=random_state,
            )
            return accuracy, conf_matrices
        # elif decoder_type == "NB":
        #     pass
        else:
            raise ValueError("Unsupported decoder_type. Must be 'SVC' or 'GBM'.")

        cv = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=random_state
        )

        accuracy = []
        conf_matrices = []
        for idxTime in range(data_arr.shape[1]):
            fold_accuracy = []
            fold_conf_matrices = []
            # get the data for the current timepoint
            # tranpose such that shape is trials x neurons
            X = data_arr[:, idxTime, :].T

            for train, test in cv.split(X, label_arr):
                XTrain = X[train, :]
                YTrain = label_arr[train]

                XTest = X[test, :]
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

                # get accuracy for the current fold
                acc = accuracy_score(YTest, predictions)
                fold_accuracy.append(acc)

                # confusion matrix
                cm = confusion_matrix(YTest, predictions)
                fold_conf_matrices.append(cm)

            # store the accuracy for the current timepoint
            accuracy.append(fold_accuracy)
            conf_matrices.append(fold_conf_matrices)

        return np.array(accuracy), np.array(conf_matrices)

    @staticmethod
    def decode_via_LSTM(
        data_arr: np.ndarray,
        label_arr: np.ndarray,
        num_folds: int,
        random_state: int = 14,
    ) -> tuple:
        input_shape = (data_arr.shape[1], data_arr.shape[2])  # timepoints x neurons
        num_classes = int(len(np.unique(label_arr)))

        cat_arr = to_categorical(label_arr - 1, num_classes=num_classes)

        accuracy = []
        confusion_matrices = []
        cv = StratifiedKFold(
            n_splits=num_folds, shuffle=True, random_state=random_state
        )
        for idx, (train, test) in enumerate(
            cv.split(np.zeros(len(label_arr)), label_arr)
        ):
            print(f"Fold: {idx + 1}")
            XTrain = data_arr[train, :, :]
            YTrain = cat_arr[train]

            XTest = data_arr[test, :, :]
            YTest = cat_arr[test]

            model = GeneralDecoder._build_LSTM(input_shape, num_classes)
            model.fit(
                XTrain,
                YTrain,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
            )

            predictions = model.predict(XTest)
            YPred = np.argmax(predictions, axis=1)
            YTrue = np.argmax(YTest, axis=1)

            accu = accuracy_score(YTrue, YPred)
            accuracy.append(accu)

            cm = confusion_matrix(YTrue, YPred)
            confusion_matrices.append(cm)

        return np.array(accuracy), np.array(confusion_matrices)

    @staticmethod
    def _build_LSTM(
        input_shape: tuple,
        num_classes: int,
        return_sequences: bool = True,
        first_layer_units: int = 128,
        second_layer_units: int = 64,
        dropout: float = 0.5,
        activation: str = "softmax",
        optimizer: str = "adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    ):
        """
        Build LSTM model.
        """
        model = Sequential()
        model.add(
            LSTM(
                units=first_layer_units,
                input_shape=input_shape,
                return_sequences=return_sequences,
            )
        )
        model.add(Dropout(rate=dropout))
        model.add(LSTM(units=second_layer_units))
        model.add(Dropout(rate=dropout))
        model.add(Dense(units=num_classes, activation=activation))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

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
