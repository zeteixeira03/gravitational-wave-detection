"""
Neural Network Model for G2Net gravitational wave detection.

Custom TensorFlow implementation.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

 
class DIYModel:
    """
    neural network for binary classification.

    Architecture:
    Loss: Binary Cross-Entropy
    Optimizer:
    """

    def __init__(self, input_dim: int = 30, hidden_dim: int = 64, hidden_dim2: int = 32,
                 learning_rate: float = 0.001, dropout_rate: float = 0.3):
        """
        initialize the DIY model.

        Parameters
        ----------
        input_dim : int
            number of input features
        hidden_dim : int
            number of hidden units in first layer
        hidden_dim2 : int
            number of hidden units in second layer
        learning_rate : float
            learning rate for gradient descent
        dropout_rate : float
            dropout rate for regularization
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # initialize weights and biases
        self.W1 = tf.Variable(tf.random.normal([input_dim, hidden_dim], stddev=np.sqrt(2.0 / input_dim)))    # using He initialization for relu
        self.b1 = tf.Variable(tf.zeros([hidden_dim]))
        self.W2 = tf.Variable(tf.random.normal([hidden_dim, hidden_dim2], stddev=np.sqrt(2.0 / hidden_dim)))
        self.b2 = tf.Variable(tf.zeros([hidden_dim2]))
        self.W3 = tf.Variable(tf.random.normal([hidden_dim2, 1], stddev=np.sqrt(2.0 / hidden_dim2)))
        self.b3 = tf.Variable(tf.zeros([1]))

        self.variables = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        perform a forward pass through the network.

        Parameters
        ----------
        x : tf.Tensor
            input tensor of shape (batch_size, input_dim)
        training : bool
            applies dropout if True

        Returns
        -------
        tf.Tensor
            output predictions of shape (batch_size, 1) with sigmoid applied
        """
        # layer 1: linear + relu + dropout
        z1 = tf.matmul(X, self.W1) + self.b1
        a1 = tf.nn.relu(z1)
        if training:
            a1 = tf.nn.dropout(a1, rate=self.dropout_rate)

        # layer 2: linear + relu + dropout
        z2 = tf.matmul(a1, self.W2) + self.b2
        a2 = tf.nn.relu(z2)
        if training:
            a2 = tf.nn.dropout(a2, rate=self.dropout_rate)

        # output layer: linear + sigmoid
        z3 = tf.matmul(a2, self.W3) + self.b3
        predictions = tf.nn.sigmoid(z3)

        return predictions

    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        compute binary cross-entropy loss.

        Parameters
        ----------
        y_true : tf.Tensor
            true labels of shape (batch_size, 1)
        y_pred : tf.Tensor
            predicted probabilities of shape (batch_size, 1)

        Returns
        -------
        tf.Tensor
            scalar loss value
        """
        # C = -[y*log(p) + (1-y)*log(1-p)]
        epsilon = 1e-7  # numerical stability
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon) # log(0) = -inf, log (1-1) = log(0)             
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

        return tf.reduce_mean(bce)

    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> float:
        """
        perform one training step (forward + backward + update).

        parameters
        ----------
        x : tf.Tensor
            input features
        y : tf.Tensor
            true labels

        Returns
        -------
        float
            loss value for this step
        """
        with tf.GradientTape() as tape:
            predictions = self.forward(x, training=True)
            loss = self.compute_loss(y, predictions)

        # gradient descent
        gradients = tape.gradient(loss, self.variables)
        for var, grad in zip(self.variables, gradients):
            var.assign_sub(self.learning_rate * grad)

        return loss.numpy()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 256,
        verbose: bool = True
    ) -> dict:
        """
        fit method

        Parameters
        ----------
        X_train : np.ndarray
            training features of shape (n_samples, n_features)
        y_train : np.ndarray
            training labels of shape (n_samples,)
        X_val : np.ndarray, optional
            validation features
        y_val : np.ndarray, optional
            validation labels
        epochs : int
            # of training epochs
        batch_size : int
            number of samples per mini-batch
        verbose : bool
            whether or not to print progress

        Returns
        -------
        dict
            training history with 'train_loss', and 'val_loss' and 'val_acc', if applicable
        """
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size  

        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val_tf = tf.convert_to_tensor(X_val, dtype=tf.float32)
            y_val_tf = tf.convert_to_tensor(y_val.reshape(-1, 1), dtype=tf.float32)

        history = {'train_loss': [], 'train_acc': [], 'train_prec': [], 'train_recall': [], 'train_spec': []}
        if has_validation:
            history['val_loss'] = []
            history['val_acc'] = []
            history['val_prec'] = []
            history['val_recall'] = []
            history['val_spec'] = []

        for epoch in range(epochs):
            # shuffle training data at start of each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # mini-batch training
            epoch_losses = []
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)

                X_batch = tf.convert_to_tensor(X_shuffled[start_idx:end_idx], dtype=tf.float32)
                y_batch = tf.convert_to_tensor(y_shuffled[start_idx:end_idx].reshape(-1, 1), dtype=tf.float32)

                batch_loss = self.train_step(X_batch, y_batch)
                epoch_losses.append(batch_loss)

            # average loss over all batches
            train_loss = np.mean(epoch_losses)
            history['train_loss'].append(train_loss)

            # get metrics, log (evaluate expects numpy arrays)
            train_metrics = self.evaluate(X_train, y_train)

            history['train_acc'].append(train_metrics['accuracy'])
            history['train_prec'].append(train_metrics['precision'])
            history['train_recall'].append(train_metrics['recall'])
            history['train_spec'].append(train_metrics['specificity'])

            # if we're doing validation
            if has_validation:
                # forward step, don't train, log loss
                val_pred = self.forward(X_val_tf)
                val_loss = self.compute_loss(y_val_tf, val_pred).numpy()
                history['val_loss'].append(val_loss)

                # get metrics, log (evaluate expects numpy arrays)
                val_metrics = self.evaluate(X_val, y_val)

                history['val_acc'].append(val_metrics['accuracy'])
                history['val_prec'].append(val_metrics['precision'])
                history['val_recall'].append(val_metrics['recall'])
                history['val_spec'].append(val_metrics['specificity'])

                # print progress
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Train Acc: {train_metrics['accuracy']:.4f} - "
                          f"Val Loss: {val_loss:.4f} - "
                          f"Val Acc: {val_metrics['accuracy']:.4f}")

            else:
                # print progress
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Train Acc: {train_metrics['accuracy']:.4f}")

        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        predict probabilities for input samples.

        Parameters
        ----------
        X : np.ndarray
            input features of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            predicted probabilities of shape (n_samples,)
        """
        X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
        predictions = self.forward(X_tf)
        
        return predictions.numpy().flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        predict binary labels for input samples.

        Parameters
        ----------
        X : np.ndarray
            input features of shape (n_samples, n_features)
        threshold : float
            classification threshold

        Returns
        -------
        np.ndarray
            predicted binary labels of shape (n_samples,)
        """
        probas = self.predict_proba(X)

        return (probas >= threshold).astype(int) # covert to bool

    def _compute_confusion_values(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """
        compute confusion matrix values from predictions and labels.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted binary labels
        y_true : np.ndarray
            True binary labels

        Returns
        -------
        dict
            Dictionary containing 'TP', 'TN', 'FP', 'FN' counts
        """
        TP = int(np.sum((y_pred == 1) & (y_true == 1)))
        TN = int(np.sum((y_pred == 0) & (y_true == 0)))
        FP = int(np.sum((y_pred == 1) & (y_true == 0)))
        FN = int(np.sum((y_pred == 0) & (y_true == 1)))
        return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    
    def confusion_matrix(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Compute confusion matrix components.

        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            True labels
        threshold : float
            Classification threshold (default: 0.5)

        Returns
        -------
        dict
            Dictionary containing 'TP', 'TN', 'FP', 'FN' counts
        """
        y_pred = self.predict(X, threshold=threshold)
        return self._compute_confusion_values(y_pred, y)
    
    def _metrics_from_confusion(self, cm: dict, n_samples: int) -> dict:
        """
        compute all metrics from confusion matrix values.

        Parameters
        ----------
        cm : dict
            Confusion matrix dict with 'TP', 'TN', 'FP', 'FN' keys
        n_samples : int
            Total number of samples

        Returns
        -------
        dict
            Dictionary containing accuracy, precision, recall, specificity, f1
        """
        TP, TN, FP, FN = cm['TP'], cm['TN'], cm['FP'], cm['FN']

        accuracy = (TP + TN) / n_samples
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1': float(f1)
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
        """
        evaluate model statistics.

        Parameters
        ----------
        X : np.ndarray
            input features
        y : np.ndarray
            true labels
        threshold : float
            Classification threshold (default: 0.5)

        Returns
        -------
        dict
            Dictionary containing:
            - 'accuracy': (TP + TN) / (TP + TN + FP + FN)
            - 'precision': TP / (TP + FP)
            - 'recall': TP / (TP + FN) [also called sensitivity]
            - 'specificity': TN / (TN + FP)
            - 'f1': 2 * (precision * recall) / (precision + recall)
        """
        cm = self.confusion_matrix(X, y, threshold=threshold)
        return self._metrics_from_confusion(cm, len(y))

    def roc_curve(self, X: np.ndarray, y: np.ndarray, n_thresholds: int = 100) -> dict:
        """
        compute ROC curve data at multiple thresholds.

        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            True labels
        n_thresholds : int
            Number of threshold points to evaluate

        Returns
        -------
        dict
            Dictionary containing:
            - 'fpr': array of false positive rates
            - 'tpr': array of true positive rates
            - 'thresholds': array of threshold values used
            - 'auc': area under the ROC curve
        """
        y_proba = self.predict_proba(X)
        thresholds = np.linspace(0, 1, n_thresholds)

        tpr_list = []
        fpr_list = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            cm = self._compute_confusion_values(y_pred, y)

            tpr = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0.0
            fpr = cm['FP'] / (cm['FP'] + cm['TN']) if (cm['FP'] + cm['TN']) > 0 else 0.0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        fpr_arr = np.array(fpr_list)
        tpr_arr = np.array(tpr_list)

        # compute AUC using trapezoidal rule
        sorted_indices = np.argsort(fpr_arr)
        fpr_sorted = fpr_arr[sorted_indices]
        tpr_sorted = tpr_arr[sorted_indices]
        auc = np.trapz(tpr_sorted, fpr_sorted)

        return {
            'fpr': fpr_arr,
            'tpr': tpr_arr,
            'thresholds': thresholds,
            'auc': float(auc)
        }

    def precision_recall_curve(self, X: np.ndarray, y: np.ndarray, n_thresholds: int = 100) -> dict:
        """
        compute precision-recall curve data at multiple thresholds.

        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            True labels
        n_thresholds : int
            Number of threshold points to evaluate

        Returns
        -------
        dict
            Dictionary containing:
            - 'precision': array of precision values
            - 'recall': array of recall values
            - 'thresholds': array of threshold values used
            - 'ap': average precision (area under PR curve)
        """
        y_proba = self.predict_proba(X)
        thresholds = np.linspace(0, 1, n_thresholds)

        precision_list = []
        recall_list = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            cm = self._compute_confusion_values(y_pred, y)

            precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 1.0  # precision=1 when no positive predictions
            recall = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0.0

            precision_list.append(precision)
            recall_list.append(recall)

        precision_arr = np.array(precision_list)
        recall_arr = np.array(recall_list)

        # compute average precision (area under PR curve, sorted by recall)
        sorted_indices = np.argsort(recall_arr)
        recall_sorted = recall_arr[sorted_indices]
        precision_sorted = precision_arr[sorted_indices]
        ap = np.trapz(precision_sorted, recall_sorted)

        return {
            'precision': precision_arr,
            'recall': recall_arr,
            'thresholds': thresholds,
            'ap': float(ap)
        }

    def save_weights(self, filepath: str) -> None:
        """
        save model weights to a file.

        Parameters
        ----------
        filepath : str
            path to save weights (will create a .npz file)
        """
        np.savez(
            filepath,
            W1=self.W1.numpy(),
            b1=self.b1.numpy(),
            W2=self.W2.numpy(),
            b2=self.b2.numpy(),
            W3=self.W3.numpy(),
            b3=self.b3.numpy()
        )

    def load_weights(self, filepath: str) -> None:
        """
        load model weights from a file.

        Parameters
        ----------
        filepath : str
            path to load weights from (.npz file)
        """
        weights = np.load(filepath)
        self.W1.assign(weights['W1'])
        self.b1.assign(weights['b1'])
        self.W2.assign(weights['W2'])
        self.b2.assign(weights['b2'])
        self.W3.assign(weights['W3'])
        self.b3.assign(weights['b3'])