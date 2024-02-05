import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def create_convolutional_model() -> models.Sequential:
    """
    Create a convolutional model for the MNIST dataset.
    :return: a convolutional model for the MNIST dataset
    """
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def visualize_training_history(history) -> None:
    """
    Visualize the training history using matplotlib.
    :param history: training history to visualize
    :return: None
    """
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend(loc='lower right')
    plt.show()


def visualize_model_predictions(num_rows: int, num_cols: int, test_images, predictions, test_labels) -> None:
    """
    Visualize the model predictions on the test dataset using matplotlib.
    Number of images to display is num_rows * num_cols.
    :param num_rows: number of rows to display
    :param num_cols: number of columns to display
    :param test_images: test images to display
    :param predictions: predictions to display
    :param test_labels: test labels to display
    :return: None
    """
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        predicted_label = np.argmax(predictions[i])
        true_label = test_labels[i]
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel("{} ({})".format(predicted_label, true_label), color=color)
    plt.show()


def main():
    # Load and split dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # Add a channels dimension (needed for convolutional model)
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]

    # Split train dataset into train and validation datasets
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2)

    model = create_convolutional_model()

    print('\nTrain the model')
    history = model.fit(train_images, train_labels, epochs=3, validation_data=(val_images, val_labels))
    visualize_training_history(history)

    print('\nEvaluate the model')
    model.evaluate(test_images,  test_labels, verbose=2)

    print('Make predictions on test data')
    # In case we prefer to have a probability distribution over the classes rather than raw logits
    # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = model.predict(test_images)

    visualize_model_predictions(5, 3, test_images, predictions, test_labels)


if __name__ == "__main__":
    main()
