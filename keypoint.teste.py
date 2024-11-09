import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv


def print_confusion_matrix(y_true, y_pred, report=True):
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    labels = [keypoint_classifier_labels[i] for i in sorted(list(set(y_true)))]
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_cmx, annot=True, fmt="g", square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()


RANDOM_SEED = 0
dataset = "model/keypoint_classifier/keypoint.csv"
model_save_path = "model/keypoint_classifier/keypoint_classifier.keras"
NUM_CLASSES = 40
max_accuracy = 0
num_iterations = 500

X_dataset = np.loadtxt(
    dataset, delimiter=",", dtype="float32", usecols=list(range(1, (21 * 2) + 1))
)
y_dataset = np.loadtxt(dataset, delimiter=",", dtype="int32", usecols=(0))

X_train, X_test, y_train, y_test = train_test_split(
    X_dataset, y_dataset, train_size=0.8, random_state=RANDOM_SEED
)


def create_model(dropout_rate=0.4, neuron_count=32):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((21 * 2,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(neuron_count, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(neuron_count // 2, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


dropouts = [0.2, 0.4, 0.6]
neurons = [16, 32, 64]

best_val_acc = 0
best_model = None

for dropout in dropouts:
    for neuron_count in neurons:
        print(f"Testando com dropout={dropout} e {neuron_count} neurônios.")

        model = create_model(dropout_rate=dropout, neuron_count=neuron_count)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            model_save_path, save_best_only=True, monitor="val_accuracy"
        )
        es_callback = tf.keras.callbacks.EarlyStopping(
            patience=10, monitor="val_accuracy", restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=128,
            validation_data=(X_test, y_test),
            callbacks=[cp_callback, es_callback],
            verbose=0,
        )

        val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
        print(f"Acurácia de Validação: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

print(f"Melhor Acurácia de Validação: {best_val_acc:.4f}")

best_model = tf.keras.models.load_model(model_save_path)

Y_pred = best_model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print(f"Melhor Acurácia de Validação: {best_val_acc:.4f}")

print_confusion_matrix(y_test, y_pred)

history = best_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback],
    verbose=1,
)


tflite_save_path = "model/keypoint_classifier/keypoint_classifier_quantized.tflite"

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

with open(tflite_save_path, "wb") as f:
    f.write(tflite_quantized_model)

print(f"Modelo TFLite quantizado salvo em: {tflite_save_path}")

interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]["index"], np.array([X_test[0]]))

interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]["index"])
