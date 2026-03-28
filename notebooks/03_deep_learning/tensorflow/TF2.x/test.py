#!/Users/caihaocui/opt/miniconda3/bin/python
# %% load the package
from os import name
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.engine.sequential import relax_input_shape

print(tf.__version__)


# %% Load the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalization the matrix to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential(
    [
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu", name="hidden_layer_1"),
        layers.Dropout(0.2),
        layers.Dense(10),
    ]
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=20)

# %%  Figure show
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(history.history["accuracy"])
ax1.set_title("model accuracy")
ax1.set_ylabel("accuracy")
ax1.set_xlabel("epoch")
ax1.legend(["train"], loc="upper left")
# plt.show()
# summarize history for loss
ax2.plot(history.history["loss"])
ax2.set_title("model loss")
ax2.set_ylabel("loss")
ax2.set_xlabel("epoch")
ax2.legend(["train"], loc="upper left")
fig.show()

model.evaluate(x_test, y_test, verbose=2)


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

probability_model(x_test[:5])
