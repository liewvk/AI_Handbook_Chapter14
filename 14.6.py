import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.keras.utils.set_random_seed(42)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.25,
    stratify=iris.target,
    random_state=42,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=0,
)

print("Training epochs:", len(history.history["loss"]))
print(f"Final training loss: {history.history['loss'][-1]:.3f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.3f}")

_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test observations:", len(y_test))
print(f"Test accuracy: {test_accuracy:.3f}")
