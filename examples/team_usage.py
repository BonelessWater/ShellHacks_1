# Example 1: Data Scientist getting training data
from data_pipeline import easy_access

# Get fraud detection data with features
train_data = easy_access.get_training_data(
    use_case="fraud_detection", sample_size=10000
)

print(f"Training data shape: {train_data.shape}")
print(f"Features: {train_data.columns.tolist()}")

# Example 2: ML Engineer using TensorFlow
import tensorflow as tf

tf_dataset = easy_access.get_model_ready_data(
    framework="tensorflow",
    dataset_name="ieee_transaction",
    batch_size=64,
    label_column="isFraud",
)

# Build model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(tf_dataset, epochs=10)

# Example 3: Data Analyst checking quality
from data_pipeline import easy_access

df = easy_access.pipeline.get_dataset("credit_card_fraud")
quality_report = easy_access.monitor.monitor_data_quality("credit_card_fraud", df)

print("Data Quality Report:")
print(f"Row count: {quality_report['row_count']}")
print(f"Issues found: {quality_report['quality_issues']}")

# Example 4: Real-time scoring
from data_pipeline.streaming import StreamProcessor


def score_transaction(data):
    # Your model scoring logic
    features = easy_access.feature_store.engineer_features(
        pd.DataFrame([data]), "transaction"
    )
    # prediction = model.predict(features)
    return {"transaction_id": data["id"], "fraud_score": 0.85}


stream_processor = StreamProcessor("vaulted-timing-473322-f9", easy_access.pipeline)
stream_processor.create_streaming_pipeline(
    "fraud_transactions", "fraud_scoring", score_transaction
)
