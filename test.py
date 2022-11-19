import tensorflow as tf
from keras.models import load_model
from asbestosis.data_generator import DataGenerator
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from keras.optimizers import Adam
from statsmodels.stats.contingency_tables import mcnemar


def get_ama(fvc, dlco):
    if not np.isnan(fvc):
        fvc = int(np.max([np.min([int(4-(fvc-49)/10), 4]), 0]))
    if not np.isnan(dlco):
        dlco = int(np.max([np.min([int(4-(dlco-44)/10), 4]), 0]))
    ama = np.nanmax([fvc, dlco])
    return ama


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def get_prediction_dict(test_pred, test_label, excel_path=None, plot=True, fig_path=None):
    nr_pul = [int(np.round(x * 3)) for x in test_label]
    d = {}
    for i, value in enumerate(nr_pul):
        if value in d:
            d[value].append(test_pred[i])
        else:
            d[value] = [test_pred[i]]
    if excel_path is not None:
        df_d = pd.DataFrame.from_dict(d, orient='index').T
        df_d = df_d.reindex(sorted(df_d.columns), axis=1)
        df_d.to_excel(excel_path)

    if plot:
        fig, ax = plt.subplots()

        boxplot_dict = ax.boxplot(
            [d[x] for x in [0, 1, 2, 3]],
            positions=[1, 1.5, 2, 2.5],
            labels=[0, 1, 2, 3],
            patch_artist=True,  # Legend isn't working with patch objects..
            widths=0.25)

        i = 0
        for b in boxplot_dict['boxes']:
            lab = ax.get_xticklabels()[i].get_text()
            print("Label property of box {0} is {1}".format(i, lab))
            b.set_label(lab)
            i += 1

        ax.set_ylim(0, 1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_tick_params(width=1)

        plt.xlabel('Number of Pulmonologists positive')
        plt.ylabel('DLCO in Model prediction')
        plt.show()
        if fig_path is not None:
            plt.savefig(fig_path)
    return d


def test():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    shape = (96, 192, 192)

    params = {'shape': shape,
              'n_classes': 2,
              'n_channels': 2,
              'label_type': 'soft'}

    filename = '/DATA/kevin/models/resnet_(96, 192, 192)_soft_2_best'
    if 'lft' in filename:
        with open('/DATA/kevin/partitions/partition_lft.pkl', 'rb') as file:
            partition, labels, lft = pickle.load(file)
    else:
        with open('/DATA/kevin//partitions/partition_soft.pkl', 'rb') as file:
            partition, labels = pickle.load(file)

    model = load_model(filename + '.h5', compile=False, custom_objects={'auc_roc': auc_roc})
    print(model.summary())

    model.compile(optimizer=Adam(learning_rate=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    test_generator = DataGenerator(partition['test'], labels, **params, augment=False, batch_size=1, shuffle=False)

    test_pred_1 = model.predict_generator(generator=test_generator,
                                          steps=len(partition['test']) // test_generator.batch_size,
                                          verbose=1)

    test_pred = np.argmax(test_pred_1, axis=-1)
    test_labels = [int(np.round(labels[k])) for k in partition['test']]
    print('Confusion Matrix')
    print(confusion_matrix(test_labels, test_pred))
    print('Classification Report')
    target_names = ['Non-Asbestosis', 'Asbestosis']
    print(classification_report(test_labels, test_pred, target_names=target_names))

    get_prediction_dict(test_pred_1[:, 1], test_labels, excel_path='/DATA/kevin/results/prediction.xlsx')


def get_confusion_matrix(name1='test_soft_1', name2='test_soft_2'):
    with open('/DATA/kevin//partitions/partition_soft.pkl', 'rb') as file:
        partition, labels = pickle.load(file)
    test_labels = [int(np.round(labels[k])) for k in partition['test']]
    with open("/DATA/kevin/pickle/{}.pkl".format(name1), 'rb') as file:
        test_1, _ = pickle.load(file)

    with open("/DATA/kevin/pickle/{}.pkl".format(name2), 'rb') as file:
        test_2, _ = pickle.load(file)

    test_1_pred = test_1[:, 1] > 0.5
    test_2_pred = test_2[:, 1] > 0.5

    y_1 = test_1_pred == test_labels
    y_2 = test_2_pred == test_labels

    conf = confusion_matrix(y_1, y_2)
    stats = mcnemar(conf)
    print(stats)

    return conf, stats


if __name__ == '__main__':
    test()
    # get_confusion_matrix('test_soft_1', 'test_soft_2')

