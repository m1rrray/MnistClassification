import matplotlib.pyplot as plt
from keras.src.optimizers import Adam
from tensorflow.python.keras.losses import categorical_crossentropy
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
у нас есть нейронная сеть, мы ее обучаем с помощью метода back propagation.
обучаем нейронку распозновать рукописные цифры, для наглядности выводятся графики
с помощью mapplotlib
"""


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])
#
y_train = to_categorical(y_train, 10)


# создаем модель нейронной сети
class MnistClassification(tf.Module):
    def __init__(self, outputs, activate="relu"):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.Variable(tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name='w'))
            self.b = tf.Variable(tf.zeros([self.outputs], dtype=tf.float32, name='b'))

            self.fl_init = True

        y = x @ self.w + self.b

        if self.activate == "relu":
            return tf.nn.relu(y)
        elif self.activate == "softmax":
            return tf.nn.softmax(y)

        return y


class SequentialModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = MnistClassification(128)
        self.layer2 = MnistClassification(10, activate="softmax")

    def __call__(self, x):
        return self.layer2(self.layer1(x))


model = SequentialModule()

cross_entropy = lambda y_true, y_pred: tf.reduce_mean(categorical_crossentropy(y_true, y_pred))
opt = Adam(learning_rate=0.001)

BATCH_SIZE = 32
EPOCHS = 10
TOTAL = x_train.shape[0]

# разбиваем нашу обучающую выборку на батчи
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

losses = []
# далее идет цикл обучения
for n in range(EPOCHS):
    loss = 0
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            f_loss = cross_entropy(y_batch, model(x_batch))

        loss += f_loss
        grads = tape.gradient(f_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    losses.append(loss.numpy())
    print(f'Epoch {n + 1}/{EPOCHS}, Loss: {loss.numpy()}')

plt.plot(range(1, EPOCHS + 1), losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

y = model(x_test)
y2 = tf.argmax(y, axis=1).numpy()
acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100
print(acc)

plt.figure()
plt.imshow(x_test[0].numpy().reshape(28, 28), cmap='gray')
plt.title(f'Ground Truth: {y_test[0]}, Predicted: {y2[0]}')
plt.show()
