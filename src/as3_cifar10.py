import copy

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras import regularizers
from keras.datasets import cifar10
from keras.layers import Dropout, BatchNormalization

seed = 7
loss_fct = "categorical_crossentropy"
opt = "adam"
metric = 'accuracy'
batch_size = 12
epochs = 40
dense_regularizer = None
mae = 0
drop = 0
bn = False

# fix random seed for reproducibility
np.random.seed(seed)

CATEGORIES = [0, 1, 8]
CATEGORIES_NAMES = ['airplane', 'automobile', 'ship']
NB_CLASS = 3
NB_FEATURES = 32 * 32 * 3


def print_to_file(title, loss, acc):
    f = open("result_cifar.txt", "a")
    f.write("\n{}   {}   {}   {}   {}   {}   {}".format(loss_fct, opt, epochs, batch_size, title, loss, acc))
    f.close()


def prepare_data():
    #  We load the data from keras
    (train_img10, train_labels10), (test_img10, test_labels10) = cifar10.load_data()

    # from the labels images, we check the number of element from our 3 selected categories
    unique, counts = np.unique(train_labels10, return_counts=True)
    counter_train = dict(zip(unique, counts))
    total_train_images = counter_train[CATEGORIES[0]] + counter_train[CATEGORIES[1]] + counter_train[CATEGORIES[2]]
    print("total number of train images : {}".format(total_train_images))

    unique, counts = np.unique(test_labels10, return_counts=True)
    counter_test = dict(zip(unique, counts))
    total_test_images = counter_test[CATEGORIES[0]] + counter_test[CATEGORIES[1]] + counter_test[CATEGORIES[2]]
    print("total number of test images : {}".format(total_test_images))

    # train_img, train_labels, test_img, test_img_labels = [], [], [], []
    train_img, train_labels = np.empty(shape=(total_train_images, 32, 32, 3)), np.empty(total_train_images)
    test_img, test_labels = np.empty(shape=(total_test_images, 32, 32, 3)), np.empty(total_test_images)

    print(train_img10[0].shape)

    ind_img, ind_lab = 0, 0
    for index in range(len(train_labels10)):
        if (train_labels10[index] in CATEGORIES):
            train_img[ind_img] = train_img10[index]
            ind_img += 1
            if train_labels10[index] == 8:
                train_labels[ind_lab] = 2
            else:
                train_labels[ind_lab] = train_labels10[index]
            ind_lab += 1
    print("size of train img : {}".format(ind_img))

    ind_img, ind_lab = 0, 0
    for index in range(len(test_labels10)):
        if (test_labels10[index] in CATEGORIES):
            test_img[ind_img] = test_img10[index]
            ind_img += 1
            if test_labels10[index] == 8:
                test_labels[ind_lab] = 2
            else:
                test_labels[ind_lab] = test_labels10[index]
            ind_lab += 1
    print("size of test img : {}".format(ind_img))

    train_img = train_img.reshape(-1, NB_FEATURES)
    # train_labels = train_img.reshape(-1, NB_FEATURES)
    test_img = test_img.reshape(-1, NB_FEATURES)
    # test_labels = train_img.reshape(-1, NB_FEATURES)

    train_img = train_img.astype("float32") / 255
    test_img = test_img.astype("float32") / 255

    # train_labels = keras.utils.to_categorical(train_labels)
    # train_labels = keras.utils.to_categorical(train_labels)

    return train_img, train_labels, test_img, test_labels, CATEGORIES


def split_training_data(data_img, data_labels):
    size_train_dataset = int(len(data_img) * 4 / 5)
    # size_train_dataset = int(len(data_img) /2)

    train_img = data_img[:size_train_dataset]
    val_img = data_img[size_train_dataset:]
    # val_img = data_img[size_train_dataset:-24]

    train_labels = data_labels[:size_train_dataset]
    val_labels = data_labels[size_train_dataset:]
    # val_labels = data_labels[size_train_dataset:-24]

    print("Test images : {} & Validation images : {}".format(len(train_img), len(val_img)))

    return train_img, train_labels, val_img, val_labels


def random_accuracy(test_labels):
    test_labels_copy = copy.copy(test_labels)
    np.random.shuffle(test_labels_copy)
    hits_array = np.array(test_labels) == np.array(test_labels_copy)
    return int((float(np.sum(hits_array)) / len(test_labels)) * 100)


def define_model(train_data):
    print("shape: ", train_data.shape[1:])

    model = models.Sequential()

    model.add(
        layers.Dense(512, activation='relu', input_shape=(train_data.shape[1:]), kernel_regularizer=dense_regularizer))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=dense_regularizer))
    if (bn): model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(layers.Dense(32, activation='relu', kernel_regularizer=dense_regularizer))
    if (bn): model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(layers.Dense(16, activation='relu', kernel_regularizer=dense_regularizer))
    if (bn): model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(layers.Dense(NB_CLASS, activation='softmax'))

    model.compile(loss=loss_fct,
                  # model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=[metric])

    return model


def plot(history, nb_epochs, title):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    first_epoch = 0
    epochs = range(first_epoch + 1, nb_epochs + 1)

    plt.plot(epochs, loss_values[first_epoch:], 'b', label='Training loss')
    plt.plot(epochs, val_loss_values[first_epoch:], 'r', label='Validation loss')
    plt.title("{} : Loss".format(title))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show(block=False)

    plt.clf()
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc_values[first_epoch:], 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc_values[first_epoch:], 'r', label='Validation Accuracy')
    plt.title("{} : Accuracy".format(title))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=False)


def main(title):
    num_predictions = 20

    train_img, train_labels, test_img, test_labels, categories_names = prepare_data()

    train_img, train_labels, val_img, val_labels = split_training_data(train_img, train_labels)

    print("We want to beat a random accuracy of : {}%".format(random_accuracy(test_labels)))

    train_labels = keras.utils.to_categorical(train_labels)
    val_labels = keras.utils.to_categorical(val_labels)
    test_labels = keras.utils.to_categorical(test_labels)

    print(train_img.shape)
    print(train_labels.shape)
    print(val_img.shape)
    print(val_labels.shape)
    print(test_img.shape)
    print(test_labels.shape)

    model = define_model(train_img)

    history = model.fit(train_img, train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(val_img, val_labels),
                        shuffle=True,
                        verbose=2)

    plot(history, epochs, title)

    print(" --- TRAINING COMPLETE --- ")

    model = define_model(train_img)

    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # we fuse the train and eval data to one data set
    train_data = np.concatenate((train_img, val_img), axis=0)
    print("final training data size : {}".format(train_data.shape))

    train_labels = np.concatenate((train_labels, val_labels), axis=0)
    print("final training labels size : {}".format(train_labels.shape))

    model.fit(train_data, train_labels, epochs=21, batch_size=12, verbose=2)

    results = model.evaluate(test_img, test_labels)

    print("\n training loss : {}".format(float(int(results[0] * 10000) / 100)))
    print("\n training accuracy : {} %".format(float(int(results[1] * 10000) / 100)))

    print_to_file(title, format(float(int(results[0] * 10000) / 100)), format(float(int(results[1] * 10000) / 100)))

    # Score trained model.
    # scores = model.evaluate(test_img, test_labels, verbose=1)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])


if __name__ == '__main__':
    f = open("result_cifar.txt", "w+")
    f.write("\n---\n")
    f.write("loss_fct   opt   epochs   batch_size   loss   accuracy")
    f.close()

    loss_fct = "hinge"
    main(loss_fct)
    loss_fct = "squared_hinge"
    main(loss_fct)
    loss_fct = "kullback_leibler_divergence"
    main(loss_fct)
    loss_fct = "categorical_crossentropy"
    main(loss_fct)

    opt = "SGD"
    main(opt)
    opt = "RMSprop"
    main(opt)
    opt = "Adagrad"
    main(opt)
    opt = "Adam"
    main(opt)

    for i in range(1, 5):
        dense_regularizer = regularizers.l2(0.1 ** i)
        main("L2_{}".format(i))

    for i in range(2, 6):
        drop = i / 10
        main("DropOut_{}".format(i / 10))

    dense_regularizer = None
    drop = 0
    bn = True
    main("BN")
