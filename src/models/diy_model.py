"""
1D CNN Model for G2Net gravitational wave detection.

Custom TensorFlow implementation using 1D convolutions on whitened signals.
This architecture processes the 3 detector signals through shared conv layers,
then concatenates features for classification.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


class DIYModel:
    """
    1D Convolutional Neural Network for binary classification of GW signals.

    Architecture:
    - Shared 1D conv layers process each detector independently
    - Features concatenated and passed through dense classifier
    - Loss: Binary Cross-Entropy
    - Optimizer: AdamW

    Input shape: (batch_size, 3, 4096) - 3 detectors, 4096 time samples
    """

    def __init__(
        self,
        n_samples: int = 4096,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.3,
        weight_decay: float = 1e-4
    ):
        """
        Initialize the DIY 1D CNN model.

        Parameters
        ----------
        n_samples : int
            Number of time samples per detector (4096 for 2s at 2048Hz).
        learning_rate : float
            Learning rate for Adam optimizer.
        dropout_rate : float
            Dropout rate for regularization.
        weight_decay : float
            L2 regularization strength for AdamW optimizer.
        """
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay

        # convolutional layer parameters: (no. filters, kernel_size, pool_size)
        # no.filters: how many convolutional blocks are in this particular layer
        # kernel_size: how many different values does each block have
        # pool_size: time dimension is shrunk by this factor, picking up the most active weight in each iteration
        self.conv_config = [
            (32, 64, 4),   # large kernel to capture low-freq patterns
            (64, 32, 4),
            (128, 16, 4),
            (256, 8, 4),
        ]

        # initialize convolutional layers
        self.conv_layers = []
        self.bn_layers = []
        for i, (filters, kernel_size, _) in enumerate(self.conv_config):
            # Conv1D weights: (kernel_size, in_channels, out_channels) - randomly assign
            in_ch = 1 if i == 0 else self.conv_config[i-1][0]
            W = tf.Variable(
                tf.random.normal([kernel_size, in_ch, filters], stddev=np.sqrt(2.0 / (kernel_size * in_ch))),
                name=f'conv{i}_W'
            )
            b = tf.Variable(tf.zeros([filters]), name=f'conv{i}_b')
            self.conv_layers.append((W, b))
            # BatchNorm params: gamma, beta, moving_mean, moving_var
            gamma = tf.Variable(tf.ones([filters]), name=f'bn{i}_gamma')
            beta = tf.Variable(tf.zeros([filters]), name=f'bn{i}_beta')
            moving_mean = tf.Variable(tf.zeros([filters]), trainable=False, name=f'bn{i}_mean')
            moving_var = tf.Variable(tf.ones([filters]), trainable=False, name=f'bn{i}_var')
            self.bn_layers.append((gamma, beta, moving_mean, moving_var))

        # GeM pooling parameter
        self.gem_p = tf.Variable(tf.constant([3.0]), name='gem_p')

        # Classifier head weights
        # After conv: 256 features per detector * 3 detectors = 768
        self.fc1_W = tf.Variable(tf.random.normal([768, 256], stddev=np.sqrt(2.0 / 768)), name='fc1_W')
        self.fc1_b = tf.Variable(tf.zeros([256]), name='fc1_b')
        self.fc1_bn = (
            tf.Variable(tf.ones([256]), name='fc1_bn_gamma'),
            tf.Variable(tf.zeros([256]), name='fc1_bn_beta'),
            tf.Variable(tf.zeros([256]), trainable=False, name='fc1_bn_mean'),
            tf.Variable(tf.ones([256]), trainable=False, name='fc1_bn_var')
        ) #output size: (batch, 256)

        self.fc2_W = tf.Variable(tf.random.normal([256, 64], stddev=np.sqrt(2.0 / 256)), name='fc2_W')
        self.fc2_b = tf.Variable(tf.zeros([64]), name='fc2_b')
        self.fc2_bn = (
            tf.Variable(tf.ones([64]), name='fc2_bn_gamma'),
            tf.Variable(tf.zeros([64]), name='fc2_bn_beta'),
            tf.Variable(tf.zeros([64]), trainable=False, name='fc2_bn_mean'),
            tf.Variable(tf.ones([64]), trainable=False, name='fc2_bn_var')
        ) # output size: (batch, 64)

        self.out_W = tf.Variable(tf.random.normal([64, 1], stddev=np.sqrt(2.0 / 64)), name='out_W')
        self.out_b = tf.Variable(tf.zeros([1]), name='out_b') # output size: (batch, 1)

        # collect all trainable variables
        self.variables = []
        for W, b in self.conv_layers:
            self.variables.extend([W, b])
        for gamma, beta, _, _ in self.bn_layers:
            self.variables.extend([gamma, beta])
        self.variables.append(self.gem_p)
        self.variables.extend([self.fc1_W, self.fc1_b, self.fc1_bn[0], self.fc1_bn[1]])
        self.variables.extend([self.fc2_W, self.fc2_b, self.fc2_bn[0], self.fc2_bn[1]])
        self.variables.extend([self.out_W, self.out_b])

        # AdamW optimizer state
        self.optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    def _batch_norm(self, x, bn_params, training):
        """Apply batch normalization."""
        gamma, beta, moving_mean, moving_var = bn_params
        eps = 1e-5

        if training:
            mean, var = tf.nn.moments(x, axes=[0, 1], keepdims=False)
            # Update moving averages
            moving_mean.assign(0.9 * moving_mean + 0.1 * mean)
            moving_var.assign(0.9 * moving_var + 0.1 * var)
        else:
            mean, var = moving_mean, moving_var

        x_norm = (x - mean) / tf.sqrt(var + eps)
        return gamma * x_norm + beta

    def _conv_block(self, x, layer_idx, training):
        """Apply conv -> batchnorm -> silu -> maxpool."""
        W, b = self.conv_layers[layer_idx]
        _, _, pool_size = self.conv_config[layer_idx]

        # Conv1D with same padding
        x = tf.nn.conv1d(x, W, stride=1, padding='SAME') + b
        # BatchNorm
        x = self._batch_norm(x, self.bn_layers[layer_idx], training)
        # SiLU
        x = x * tf.nn.sigmoid(x)
        # MaxPool
        x = tf.nn.max_pool1d(x, ksize=pool_size, strides=pool_size, padding='SAME')
        return x

    def _gem_pool(self, x):
        """Generalized Mean Pooling."""
        p = tf.clip_by_value(self.gem_p, 1.0, 10.0)
        eps = 1e-6
        x = tf.clip_by_value(x, eps, 1e10)
        x_pow = tf.pow(x, p)
        mean_pow = tf.reduce_mean(x_pow, axis=1)
        return tf.pow(mean_pow, 1.0 / p)

    def _process_detector(self, x, training):
        """Process a single detector through conv layers."""
        # x shape: (batch, time_steps), but must be (batch, time, 1) for tf
        x = tf.expand_dims(x, axis=-1)

        # Apply conv blocks
        for i in range(len(self.conv_layers)):
            x = self._conv_block(x, i, training)

        # GeM pooling: (batch, reduced_time, 256) -> (batch, 256)
        x = self._gem_pool(x)
        return x

    def forward(self, X: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        X : tf.Tensor
            Input tensor of shape (batch_size, 3, n_samples).
        training : bool
            Applies dropout if True.

        Returns
        -------
        tf.Tensor
            Output predictions of shape (batch_size, 1) with sigmoid applied.
        """
        # split detectors: X shape is (batch, 3, 4096)
        h1 = X[:, 0, :]  # (batch, 4096)
        l1 = X[:, 1, :]
        v1 = X[:, 2, :]

        # process each detector through shared conv layers
        h1_feat = self._process_detector(h1, training)  # (batch, 256)
        l1_feat = self._process_detector(l1, training)
        v1_feat = self._process_detector(v1, training)

        # concatenate features from all detectors
        combined = tf.concat([h1_feat, l1_feat, v1_feat], axis=-1)  # (batch, 768)

        # Classifier head
        # FC1
        x = tf.matmul(combined, self.fc1_W) + self.fc1_b        # matrix multiplication reduces time dimension (3x256) -> 256
        x = self._batch_norm(tf.expand_dims(x, 1), self.fc1_bn, training)
        x = tf.squeeze(x, 1)
        x = x * tf.nn.sigmoid(x)  # SiLU
        if training:
            x = tf.nn.dropout(x, rate=self.dropout_rate)

        # FC2
        x = tf.matmul(x, self.fc2_W) + self.fc2_b               # matrix multiplication reduces time dimension 256 -> 64
        x = self._batch_norm(tf.expand_dims(x, 1), self.fc2_bn, training)
        x = tf.squeeze(x, 1)
        x = x * tf.nn.sigmoid(x)  # SiLU
        if training:
            x = tf.nn.dropout(x, rate=self.dropout_rate)

        # Output
        x = tf.matmul(x, self.out_W) + self.out_b               # matrix multiplication reduces time dimension 64 -> 1
        predictions = tf.nn.sigmoid(x)

        return predictions

    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute binary cross-entropy loss.

        Parameters
        ----------
        y_true : tf.Tensor
            True labels of shape (batch_size, 1).
        y_pred : tf.Tensor
            Predicted probabilities of shape (batch_size, 1).

        Returns
        -------
        tf.Tensor
            Scalar loss value.
        """
        # C = -[y*log(p) + (1-y)*log(1-p)]
        epsilon = 1e-7  # numerical stability
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon) # log(0) = -inf, log (1-1) = log(0)             
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

        return tf.reduce_mean(bce)

    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> float:
        """
        Perform one training step (forward + backward + update).

        Parameters
        ----------
        x : tf.Tensor
            Input signals of shape (batch, 3, n_samples).
        y : tf.Tensor
            True labels.

        Returns
        -------
        float
            Loss value for this step.
        """
        with tf.GradientTape() as tape:
            predictions = self.forward(x, training=True)
            loss = self.compute_loss(y, predictions)

        # AdamW optimizer update
        gradients = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables))

        return loss.numpy()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ) -> dict:
        """
        Train the model on training data.

        Parameters
        ----------
        X_train : np.ndarray
            Training signals of shape (n_samples, 3, n_time_samples).
        y_train : np.ndarray
            Training labels of shape (n_samples,).
        X_val : np.ndarray, optional
            Validation signals.
        y_val : np.ndarray, optional
            Validation labels.
        epochs : int
            Number of training epochs.
        batch_size : int
            Number of samples per mini-batch (smaller for CNN due to memory).
        verbose : bool
            Whether to print progress.

        Returns
        -------
        dict
            Training history with 'train_loss', and 'val_loss' and 'val_acc' if applicable.
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
            # shuffle training data at start of each epoc
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

            # get metrics, log
            train_metrics = self.evaluate(X_train, y_train)

            history['train_acc'].append(train_metrics['accuracy'])
            history['train_prec'].append(train_metrics['precision'])
            history['train_recall'].append(train_metrics['recall'])
            history['train_spec'].append(train_metrics['specificity'])

            if has_validation:
                # forward step, don't train, log loss
                val_pred = self.forward(X_val_tf)
                val_loss = self.compute_loss(y_val_tf, val_pred).numpy()
                history['val_loss'].append(val_loss)

                # get metrics, log
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

    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Predict probabilities for input samples.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, 3, n_time_samples).
        batch_size : int
            Batch size for inference to avoid OOM on large inputs.

        Returns
        -------
        np.ndarray
            Predicted probabilities of shape (n_samples,).
        """
        n_samples = X.shape[0]

        # small input: process all at once
        if n_samples <= batch_size:
            X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
            predictions = self.forward(X_tf)
            return predictions.numpy().flatten()

        # large input: process in batches
        all_predictions = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            X_batch = tf.convert_to_tensor(X[start_idx:end_idx], dtype=tf.float32)
            batch_pred = self.forward(X_batch)
            all_predictions.append(batch_pred.numpy())

        return np.concatenate(all_predictions, axis=0).flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels for input samples. 1 = BH merger, 0 = not

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).
        threshold : float
            Classification threshold.

        Returns
        -------
        np.ndarray
            Predicted binary labels of shape (n_samples,).
        """
        probas = self.predict_proba(X)

        return (probas >= threshold).astype(int)

    def _compute_confusion_values(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Compute confusion matrix values from predictions and labels.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted binary labels.
        y_true : np.ndarray
            True binary labels.

        Returns
        -------
        dict
            Dictionary containing 'TP', 'TN', 'FP', 'FN' counts.
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
            Input features.
        y : np.ndarray
            True labels.
        threshold : float
            Classification threshold.

        Returns
        -------
        dict
            Dictionary containing 'TP', 'TN', 'FP', 'FN' counts.
        """
        y_pred = self.predict(X, threshold=threshold)
        return self._compute_confusion_values(y_pred, y)
    
    def _metrics_from_confusion(self, cm: dict, n_samples: int) -> dict:
        """
        Compute all metrics from confusion matrix values.

        Parameters
        ----------
        cm : dict
            Confusion matrix dict with 'TP', 'TN', 'FP', 'FN' keys.
        n_samples : int
            Total number of samples.

        Returns
        -------
        dict
            Dictionary containing accuracy, precision, recall, specificity, f1.
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
        Evaluate model statistics.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True labels.
        threshold : float
            Classification threshold.

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
        Compute ROC curve data at multiple thresholds.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True labels.
        n_thresholds : int
            Number of threshold points to evaluate.

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
        auc = np.trapezoid(tpr_sorted, fpr_sorted)

        return {'fpr': fpr_arr, 'tpr': tpr_arr, 'thresholds': thresholds, 'auc': float(auc)}

    def precision_recall_curve(self, X: np.ndarray, y: np.ndarray, n_thresholds: int = 100) -> dict:
        """
        Compute precision-recall curve data at multiple thresholds.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True labels.
        n_thresholds : int
            Number of threshold points to evaluate.

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
        ap = np.trapezoid(precision_sorted, recall_sorted)

        return {'precision': precision_arr, 'recall': recall_arr, 'thresholds': thresholds, 'ap': float(ap)}

    def save_weights(self, filepath: str) -> None:
        """
        Save model weights to a file.

        Parameters
        ----------
        filepath : str
            Path to save weights (will create a .npz file).
        """
        weights_dict = {}

        # Conv layers
        for i, (W, b) in enumerate(self.conv_layers):
            weights_dict[f'conv{i}_W'] = W.numpy()
            weights_dict[f'conv{i}_b'] = b.numpy()

        # BatchNorm layers
        for i, (gamma, beta, moving_mean, moving_var) in enumerate(self.bn_layers):
            weights_dict[f'bn{i}_gamma'] = gamma.numpy()
            weights_dict[f'bn{i}_beta'] = beta.numpy()
            weights_dict[f'bn{i}_mean'] = moving_mean.numpy()
            weights_dict[f'bn{i}_var'] = moving_var.numpy()

        # GeM parameter
        weights_dict['gem_p'] = self.gem_p.numpy()

        # FC layers
        weights_dict['fc1_W'] = self.fc1_W.numpy()
        weights_dict['fc1_b'] = self.fc1_b.numpy()
        weights_dict['fc1_bn_gamma'] = self.fc1_bn[0].numpy()
        weights_dict['fc1_bn_beta'] = self.fc1_bn[1].numpy()
        weights_dict['fc1_bn_mean'] = self.fc1_bn[2].numpy()
        weights_dict['fc1_bn_var'] = self.fc1_bn[3].numpy()

        weights_dict['fc2_W'] = self.fc2_W.numpy()
        weights_dict['fc2_b'] = self.fc2_b.numpy()
        weights_dict['fc2_bn_gamma'] = self.fc2_bn[0].numpy()
        weights_dict['fc2_bn_beta'] = self.fc2_bn[1].numpy()
        weights_dict['fc2_bn_mean'] = self.fc2_bn[2].numpy()
        weights_dict['fc2_bn_var'] = self.fc2_bn[3].numpy()

        weights_dict['out_W'] = self.out_W.numpy()
        weights_dict['out_b'] = self.out_b.numpy()

        np.savez(filepath, **weights_dict)

    def load_weights(self, filepath: str) -> None:
        """
        Load model weights from a file.

        Parameters
        ----------
        filepath : str
            Path to load weights from (.npz file).
        """
        weights = np.load(filepath)

        # Conv layers
        for i, (W, b) in enumerate(self.conv_layers):
            W.assign(weights[f'conv{i}_W'])
            b.assign(weights[f'conv{i}_b'])

        # BatchNorm layers
        for i, (gamma, beta, moving_mean, moving_var) in enumerate(self.bn_layers):
            gamma.assign(weights[f'bn{i}_gamma'])
            beta.assign(weights[f'bn{i}_beta'])
            moving_mean.assign(weights[f'bn{i}_mean'])
            moving_var.assign(weights[f'bn{i}_var'])

        # GeM parameter
        self.gem_p.assign(weights['gem_p'])

        # FC layers
        self.fc1_W.assign(weights['fc1_W'])
        self.fc1_b.assign(weights['fc1_b'])
        self.fc1_bn[0].assign(weights['fc1_bn_gamma'])
        self.fc1_bn[1].assign(weights['fc1_bn_beta'])
        self.fc1_bn[2].assign(weights['fc1_bn_mean'])
        self.fc1_bn[3].assign(weights['fc1_bn_var'])

        self.fc2_W.assign(weights['fc2_W'])
        self.fc2_b.assign(weights['fc2_b'])
        self.fc2_bn[0].assign(weights['fc2_bn_gamma'])
        self.fc2_bn[1].assign(weights['fc2_bn_beta'])
        self.fc2_bn[2].assign(weights['fc2_bn_mean'])
        self.fc2_bn[3].assign(weights['fc2_bn_var'])

        self.out_W.assign(weights['out_W'])
        self.out_b.assign(weights['out_b'])