import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class TensorFlowDataPipeline:
    """TensorFlow-specific data pipeline"""

    def __init__(self, data_pipeline: Any, feature_store: Any):
        # Use Any for types to avoid import-time forward reference errors
        self.pipeline = data_pipeline
        self.feature_store = feature_store

    def create_tf_dataset(
        self,
        dataset_name: str,
        batch_size: int = 32,
        feature_columns: List[str] = None,
        label_column: str = "isFraud",
        cache: bool = True,
    ) -> tf.data.Dataset:
        """Create TensorFlow dataset from BigQuery"""

        # Get data
        df = self.pipeline.get_dataset(dataset_name, columns=feature_columns)

        # Engineer features
        df = self.feature_store.engineer_features(df, "transaction")

        # Separate features and labels
        if label_column in df.columns:
            labels = df[label_column].to_numpy()
            features_df = df.drop(columns=[label_column])
        else:
            labels = None
            features_df = df.copy()

        # Drop datetime-like columns and convert booleans to numeric
        if not features_df.empty:
            dt_cols = features_df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
            if dt_cols:
                features_df = features_df.drop(columns=dt_cols)

            # Convert booleans to integers
            bool_cols = features_df.select_dtypes(include=["bool"]).columns.tolist()
            for c in bool_cols:
                features_df[c] = features_df[c].astype(int)

            # Convert categorical columns to numeric using one-hot encoding and fill NaNs
            features_df = pd.get_dummies(features_df).fillna(0)
        
        features = features_df

        # Create TF dataset
        if labels is not None:
            # Ensure numeric dtype for ML frameworks
            features = np.asarray(features, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.float32)
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(np.asarray(features, dtype=np.float32))

        # Apply transformations
        if cache:
            dataset = dataset.cache()

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def create_image_dataset(
        self,
        dataset_name: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
    ) -> tf.data.Dataset:
        """Create TensorFlow dataset for images from GCS"""

        # Get image metadata
        df = self.pipeline.get_dataset(dataset_name)

        def load_and_preprocess_image(path, label):
            # Load image from GCS
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, image_size)
            image = tf.cast(image, tf.float32) / 255.0
            return image, label

        # Create dataset from paths
        paths = df["image_gcs_path"].values
        labels = (
            df["is_fraudulent"].values
            if "is_fraudulent" in df.columns
            else np.zeros(len(df))
        )
        dataset = tf.data.Dataset.from_tensor_slices((paths, np.asarray(labels, dtype=np.float32)))
        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


class PyTorchDataPipeline:
    """PyTorch-specific data pipeline"""

    def __init__(self, data_pipeline: Any, feature_store: Any):
        self.pipeline = data_pipeline
        self.feature_store = feature_store

    class FraudDataset(Dataset):
        """PyTorch Dataset for fraud detection"""

        def __init__(self, df: pd.DataFrame, label_column: str = "isFraud"):
            # Coerce to numeric arrays to avoid object dtype problems
            proc = df.copy()
            if label_column in proc.columns:
                labels = proc[label_column].to_numpy(dtype=np.float32)
                proc = proc.drop(columns=[label_column])
            else:
                labels = None

            # Drop datetime columns and convert booleans
            dt_cols = proc.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
            if dt_cols:
                proc = proc.drop(columns=dt_cols)
            bool_cols = proc.select_dtypes(include=["bool"]).columns.tolist()
            for c in bool_cols:
                proc[c] = proc[c].astype(int)

            proc = pd.get_dummies(proc).fillna(0)

            self.features = proc.to_numpy(dtype=np.float32)
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            if self.labels is not None:
                return torch.FloatTensor(self.features[idx]), torch.FloatTensor(
                    [self.labels[idx]]
                )
            return torch.FloatTensor(self.features[idx])

    class ImageDataset(Dataset):
        """PyTorch Dataset for image forgery detection"""

        def __init__(self, df: pd.DataFrame, storage_client, transform=None):
            self.df = df
            self.storage_client = storage_client
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]

            # Parse GCS path
            gcs_path = row["image_gcs_path"]
            bucket_name = gcs_path.split("/")[2]
            blob_path = "/".join(gcs_path.split("/")[3:])

            # Load image from GCS
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            image_bytes = blob.download_as_bytes()

            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            if self.transform:
                image = self.transform(image)

            label = row["is_fraudulent"] if "is_fraudulent" in row else 0

            return image, torch.tensor(label, dtype=torch.float32)

    def create_dataloader(
        self,
        dataset_name: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> DataLoader:
        """Create PyTorch DataLoader"""

        # Get and process data
        df = self.pipeline.get_dataset(dataset_name)
        df = self.feature_store.engineer_features(df, "transaction")

        # Create dataset
        dataset = self.FraudDataset(df)

        # Create dataloader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

        return dataloader
