import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras import models
from keras import regularizers
import sys
# Hyper parameters :
from keras.layers import Dropout, BatchNormalization

seed = 7
loss_fct = "mse"
opt = "rmsprop"
metric = 'mae'
num_epochs = 50
batch_size = 1
dense_regularizer = regularizers.l2(0.0001)
mae = 0
drop = 0
bn = False

# fix random seed for reproducibility
np.random.seed(seed)

def print_to_file(mae):
    f = open("result_crime.txt", "a")
    f.write("\n{}   {}   {}   {}   {}   {}".format(loss_fct, opt, metric, num_epochs, batch_size, mae))
    f.close()


def shuffle_unison(array_1, array_2):
    # fuse the two array into one
    array_3 = np.c_[array_1.reshape(len(array_1), -1), array_2.reshape(len(array_2), -1)]

    # shuffle the resulting array
    np.random.shuffle(array_3)

    # split the arrays like the originals and return it
    return array_3[:, :array_1.size // len(array_1)].reshape(array_1.shape), array_3[:,
                                                                             array_1.size // len(array_1):].reshape(
        array_2.shape)


def split_data(data, labels):
    split_test = int(len(data) * 4 / 5)

    train_data = data[:split_test]
    train_labels = labels[:split_test]
    test_data = data[split_test:]
    test_labels = labels[split_test:]

    split_eval = int(len(train_data) * 4 / 5)

    val_data = train_data[split_eval:]
    val_labels = train_labels[split_eval:]
    train_data = train_data[:split_eval]
    train_labels = train_labels[:split_eval]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def load_crime_data(path_data):
    file = open(path_data, "r")

    compteur = 0

    train_data, train_labels = np.zeros((1994, 122)), np.zeros(1994)
    ind_train, ind_lab = 0, 0
    for line in file:
        array_line = line.split(",")
        # print("size ",len(array_line))

        # Information about the state, not use for prediction
        info = array_line[:5]
        # print("info : {}".format(info))

        # Replace missing values by -1
        for i in range(len(array_line)):
            if i >= 5 and array_line[i] == "?":
                array_line[i] = -1.0

        # All data are already normalized
        data = array_line[5:127]
        train_data[ind_train] = data
        ind_train += 1

        # the labels are the last value
        label = float(array_line[127])
        train_labels[ind_lab] = label
        ind_lab += 1

    print("Train data size : ", train_data.shape)
    # print("Train data : ", train_data[0])
    print("Train labels size : ", train_labels.shape)
    print("Train labels : ", train_labels[:9])

    # We need to shuffle the data, as the original data was sorted
    # we shuffle the two list together as keeping the same index is important
    print(" --- shuffling --- ")
    train_data, train_labels = shuffle_unison(train_data, train_labels)

    print("Train data size : ", train_data.shape)
    # print("Train data : ", train_data[0])
    print("Train labels size : ", train_labels.shape)
    print("Train labels : ", train_labels[:9])

    return split_data(train_data, train_labels)


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def build_model(train_data):
    model = models.Sequential()

    model.add(
        layers.Dense(64, activation="relu", input_shape=(train_data.shape[1:]), kernel_regularizer=dense_regularizer))
    if bn:model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(layers.Dense(32, activation="relu", kernel_regularizer=dense_regularizer))
    if bn:model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(layers.Dense(1, kernel_regularizer=dense_regularizer))

    model.compile(optimizer=opt, loss=loss_fct, metrics=[metric])
    return model


def main(path_data, plt_title):
    train_data, train_labels, my_val_data, my_val_labels, test_data, test_labels = load_crime_data(path_data)

    model = build_model(train_data)

    # --- K-FOLD CROSS VALIDATION
    k = 4
    num_val_samples = len(train_data) // k
    all_scores = []
    all_mae_history = []

    for i in range(k):
        print("processing fold #{}".format(i + 1))

        # prepare validation fold
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_labels = train_labels[i * num_val_samples: (i + 1) * num_val_samples]

        # concacenate training folds
        partial_train_data = np.concatenate(
            (train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]), axis=0)
        partial_train_labels = np.concatenate(
            (train_labels[:i * num_val_samples],
             train_labels[(i + 1) * num_val_samples:]), axis=0)

        model = build_model(train_data)

        history = model.fit(
            partial_train_data,
            partial_train_labels,
            validation_split=0.33,
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=0
        )

        mae_history = history.history['val_mean_absolute_error']
        all_mae_history.append(mae_history)

        val_mse, val_mae = model.evaluate(val_data, val_labels, verbose=0)
        all_scores.append(val_mae)

    average_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]

    smooth_mae_history = smooth_curve(average_mae_history[0:])

    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.title(plt_title)
    plt.show()

    print("all_scores :", all_scores)

    print("Mean Absolute Error : {}".format(np.mean(all_scores)))

    print("Absolute validation error per fold : ")
    for i in range(k):
        print("fold #{0} : {1}".format(i, all_scores[i]))


    global mae
    mae = np.mean(all_scores)
    print_to_file(mae)

    # ---- TESTING -----
    #
    # print("testing the model")
    # model = build_model(train_data)
    #
    # model.fit(train_data, train_labels, epochs=40, batch_size=1, verbose=2)

    # test_mse_score, test_mae_score = model.evaluate(test_data, test_labels)
    #
    # print("FINAL MSE : {}".format(test_mse_score))
    # print("FINAL MAE : {}".format(test_mae_score))


if __name__ == '__main__':
    nb_arg = len(sys.argv) - 1
    if nb_arg >= 1 :
        path_data=sys.argv[1]
    else:
        path_data = None
        print("ERROR please give data path")

    f = open("result_crime.txt", "w+")
    f.write("\n---\n")
    f.write("loss   opt   metric   epochs   batch_size   mae")
    f.close()

    loss_fct = "mean_squared_error"
    main(path_data, loss_fct)
    loss_fct = "mean_absolute_error"
    main(path_data, loss_fct)
    loss_fct = "logcosh"
    main(path_data, loss_fct)
    loss_fct = tf.losses.huber_loss
    main(path_data, loss_fct)

    opt = "SGD"
    main(path_data, opt)
    opt = "RMSprop"
    main(path_data, opt)
    opt = "Adagrad"
    main(path_data, opt)
    opt = "Adam"
    main(path_data, opt)

    for i in range(1,5):
        dense_regularizer = regularizers.l2(0.1**i)
        main(path_data, "L2_{}".format(i))

    for i in range(2,6):
        drop = i/10
        main(path_data, "DropOut_{}".format(i/10))

    dense_regularizer = None
    drop = 0
    bn = True
    main(path_data, "BN")