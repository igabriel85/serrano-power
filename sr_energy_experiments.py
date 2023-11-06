"""
Serrano Energy consumption experiments
Goals:
- Run predictive models and measure energy consumption
    - Run predictive models and measure energy consumption with different
    - Run predicitive models on different datasets and measure energy consumption
"""
import os
# limit GPU allocation
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import yaml
import sys
import time
import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer, classification_report, accuracy_score, jaccard_score
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.metrics import classification_report_imbalanced
import xgboost as xgb
import lightgbm as lgm
from catboost import CatBoostClassifier
import tensorflow as tf
from tab2img.converter import Tab2Img
from subprocess import check_output
# from sklearn.externals import joblib
from joblib import dump, load
from skimage.io import imsave
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# instrumenting code for power consumption
try:
    from pyJoules.device import DeviceFactory
    from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain
    from pyJoules.device.nvidia_device import NvidiaGPUDomain
    from pyJoules.energy_meter import EnergyMeter
    try:
        domains = [RaplPackageDomain(0), RaplCoreDomain(0), NvidiaGPUDomain(0)]
    except Exception as inst:
        print(f"Missing RAPL Domain with {type(inst)} and {inst.args}")
        domains = [RaplPackageDomain(0),
                   NvidiaGPUDomain(0)]
    devices = DeviceFactory.create_devices(domains)
    meter = EnergyMeter(devices)
except:
    print("pyJoules not installed")
    power_meter = 0
    pass

# TODO check deprecation warning xgboost
import warnings
warnings.filterwarnings('ignore')

# add env variable
train_dir = os.getenv("TRAIN_DIR", default="data/train")

def print_verbose(msg, verbose=1):
    # verbose = os.getenv("SER_VERBOSE", default=False)
    # print(verbose)
    if verbose:
        print(msg)

def custom_scoring_reporting(y_pred,
                             y,
                             definitions,
                             prefix,
                             model_dir):
    """
    Custom function for handling scoring and reporting
    :param y_pred: model predictions
    :param y: ground truth
    :param definitions: variable class definitions (factorize)
    :param prefix: prefix to saved files and images
    :return: 0
    """
    print_verbose("Accuracy score is: {}".format(accuracy_score(y, y_pred)))
    print_verbose("Ballanced accuracy score is: {}".format(balanced_accuracy_score(y, y_pred)))
    print_verbose("Jaccard score (micro): {}".format(jaccard_score(y, y_pred, average='micro')))
    print_verbose("Jaccard score (macro): {}".format(jaccard_score(y, y_pred, average='macro')))
    print_verbose("Jaccard score (weighted): {}".format(jaccard_score(y, y_pred, average='weighted')))


    print_verbose("Full classification report")
    print_verbose(classification_report(y, y_pred, target_names=definitions))
    report = classification_report(y, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    classification_rep_name = "{}_classification_rep_best.csv".format(prefix)
    df_classification_report.to_csv(os.path.join(model_dir,classification_rep_name), index=False)


    print_verbose("Imbalanced Classification report")
    print_verbose(classification_report_imbalanced(y, y_pred, target_names=definitions))
    imb_report = classification_report_imbalanced(y, y_pred, target_names=definitions, output_dict=True)
    df_imb_classification_report = pd.DataFrame(imb_report).transpose()
    classification_imb_rep_name = "{}_imb_classification_rep_best.csv".format(prefix)
    df_imb_classification_report.to_csv(os.path.join(model_dir,classification_imb_rep_name), index=False)


def save_valid_curve_scores(viz,
                            param,
                            param_range,
                            fname,
                            model_dir):
    """
    :param viz: yellowbrick viz object
    :param param: parameter name used in validation curve
    :param fname: mL_method used
    :return: dataframe with crossvalidation scores for param and param range set by user
    """
    columns = []
    for f in range(viz.cv.n_splits):
        columns.append(f"Fold{f}")
    scores = viz.test_scores_
    df_scores = pd.DataFrame(scores, index=param_range, columns=columns)
    df_scores.to_csv(os.path.join(model_dir, f"{fname}_{param}_validationcurve_scores.csv"))
    return df_scores

def check_data_folders(conf):
    print("Setting paths and datasets")
    # Checking if directory exists for data, modells and processed
    # print(check_output(["ls", train_dir]).decode("utf8"))

    # Check if exp directory exists
    exp_dir = os.getenv("EXP_DIR", default=f"data/train/{conf['experiment']}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    model_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return exp_dir, model_dir


def load_data(index_col='time'):
    df_anomaly = pd.read_csv(os.path.join(train_dir, "df_anomaly.csv"))
    df_audsome = pd.read_csv(os.path.join(train_dir, "df_audsome.csv"))
    df_clean = pd.read_csv(os.path.join(train_dir, "df_clean_single.csv"))
    df_clean_audsome = pd.read_csv(os.path.join(train_dir, "df_clean_ausdome_single.csv"))

    # Set index as time
    df_anomaly.set_index(index_col, inplace=True)
    df_audsome.set_index(index_col, inplace=True)
    df_clean.set_index(index_col, inplace=True)
    df_clean_audsome.set_index(index_col, inplace=True)

    # Dirty fix for df_clean_audsome
    df_clean_audsome.drop(['target_cpu_master',
                           'target_mem_master', 'target_copy_master', 'target_ddot_master'], axis=1, inplace=True)
    print_verbose("Dataset chosen ...")
    data = df_anomaly
    # drop_col = ['t1','t2','t3','t4']
    print_verbose("Remove unwanted columns ...")
    # print("Shape before drop: {}".format(data.shape))
    # data.drop(drop_col, axis=1, inplace=True)
    # print("Shape after drop: {}".format(data.shape))
    return df_anomaly, df_audsome, df_clean, df_clean_audsome


def dnn_serrano(
        optimizer='adam', #adam, adagrad, sgd
        learning_r=0.01,
        kernel_init='he_uniform',
        layer_1=20,
        layer_2=40,
        layer_3=40,
        layer_0=100,
        drop=0.1,
        loss='categorical_crossentropy',
        activation_1='relu', # elu, selu
        out_activation='sigmoid',
        input_dim=89,
        outp_dim=5
):
    # y_oh = pd.get_dummies(y, prefix='target')
    # n_inputs, n_outputs = X.shape[1], len(y_oh.nunique())
    n_inputs, n_outputs = input_dim, outp_dim
    x_in = tf.keras.layers.Input(shape=n_inputs)
    if layer_0:
        x = tf.keras.layers.Dense(layer_0, input_dim=n_inputs, kernel_initializer=kernel_init, activation=activation_1)(x_in)
        x = tf.keras.layers.Dropout(drop)(x)
    if layer_1:
        x = tf.keras.layers.Dense(layer_1, input_dim=n_inputs, kernel_initializer=kernel_init, activation=activation_1)(x)
        x = tf.keras.layers.Dropout(drop)(x)
    if layer_2:
        x = tf.keras.layers.Dense(layer_2, input_dim=n_inputs, kernel_initializer=kernel_init, activation=activation_1)(x)
        x = tf.keras.layers.Dropout(drop)(x)
    if layer_3:
        x = tf.keras.layers.Dense(layer_3, input_dim=n_inputs, kernel_initializer=kernel_init, activation=activation_1)(x)
        x = tf.keras.layers.Dropout(drop)(x)
    x_out = tf.keras.layers.Dense(n_outputs, activation=out_activation)(x)
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_r)
    if optimizer == 'adagrad':
        opt = tf.keras.optimizers.Adagrad(learning_rate=learning_r)
    if optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_r)
    if optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_r)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model

def build_model_v2(num_classes,
                   layer_1=512,
                   layer_2=256,
                   layer_3=128,
                   drop_1=0.0,
                   drop_2=0.0,
                   drop_3=0.0,
                   drop_4=0.0,
                   drop_5=0.0,
                   drop_6=0.0,
                   activation_1='relu',
                   activation_2='relu',
                   activation_3='relu',
                   activation_4='relu',
                   activation_5='relu',
                   activation_6='relu',
                   optimizer='adam',
                   learning_r='0.001',
                   input_c1=8,
                   input_c2=8,
                   ):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(input_c1, input_c2, 1)))
    if drop_1:
        model.add(tf.keras.layers.Dropout(drop_1))
    model.add(tf.keras.layers.MaxPooling2D((1, 1)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=activation_1))
    if drop_2:
        model.add(tf.keras.layers.Dropout(drop_2))
    model.add(tf.keras.layers.MaxPooling2D((1, 1)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=activation_2))
    if drop_3:
        model.add(tf.keras.layers.Dropout(drop_3))
    model.add(tf.keras.layers.Flatten())
    if layer_1:
        model.add(tf.keras.layers.Dense(layer_1, activation=activation_3))
    if layer_2:
        model.add(tf.keras.layers.Dense(layer_2, activation=activation_4))
    if drop_4:
        model.add(tf.keras.layers.Dropout(drop_4))
    if layer_3:
        model.add(tf.keras.layers.Dense(layer_3, activation=activation_5))
    if drop_5:
        model.add(tf.keras.layers.Dropout(drop_5))
    model.add(tf.keras.layers.Dense(64, activation=activation_6))
    if drop_6:
        model.add(tf.keras.layers.Dropout(drop_6))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_r)
    if optimizer == 'adagrad':
        opt = tf.keras.optimizers.Adagrad(learning_rate=learning_r)
    if optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_r)
    if optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_r)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model


def generate_images(images,
                    train_data_index,
                    test_data_index,
                    labels,
                    train_dir=train_dir):
    input_dir = os.path.join(train_dir, 'input')
    if os.path.exists(input_dir):
        import shutil
        shutil.rmtree(input_dir)
        print("Removed existing training dir")
        os.mkdir(input_dir)
    else:
        os.mkdir(input_dir)

    for ind, label in enumerate(labels):
        if ind in train_data_index:
            train_dir = os.path.join(input_dir, 'training')
            label_dir = os.path.join(train_dir, str(label))
            if os.path.exists(label_dir):
                pass
            else:
                os.makedirs(label_dir)
        elif ind in test_data_index:
            val_dir = os.path.join(input_dir, 'validation')
            label_dir = os.path.join(val_dir, str(label))
            if os.path.exists(label_dir):
                pass
            else:
                os.makedirs(label_dir)
        else:
            print(ind)
            import sys
            sys.exit()
        im_name = f'data_{ind}.png'
        im_path = os.path.join(label_dir, im_name)
        imsave(im_path,images[ind].astype(np.uint8)) # fixed warnings
    return train_dir, val_dir


def select_method(exp_method_conf, params=None):
    if 'AdaBoostClassifier' in exp_method_conf['method']:
        if params is None:
            clf = AdaBoostClassifier(**exp_method_conf['params'])
        else:
            clf = AdaBoostClassifier(**params)
        print_verbose("Method chosen: {}".format(clf))
        print_verbose("Method params: {}".format(exp_method_conf['params']))
    elif 'RandomForestClassifier' in exp_method_conf['method']:
        if params is None:
            clf = RandomForestClassifier(**exp_method_conf['params'])
        else:
            clf = RandomForestClassifier(**params)

    elif 'XGBoost' in exp_method_conf['method']:
        if params is None:
            clf = xgb.XGBClassifier(**exp_method_conf['params'])
        else:
            clf = xgb.XGBClassifier(**params)

    elif 'LightGBM' in exp_method_conf['method']:
        if params is None:
            clf = lgm.sklearn.LGBMClassifier(**exp_method_conf['params'])
        else:
            clf = lgm.sklearn.LGBMClassifier(**params)

    elif 'CatBoost' in exp_method_conf['method']:
        if params is None:
            clf = CatBoostClassifier(**exp_method_conf['params'])
        else:
            clf = CatBoostClassifier(**params)
    elif 'dnn' in exp_method_conf['method']:
        if params is None:
            clf = dnn_serrano(**exp_method_conf['params'])
        else:
            clf = dnn_serrano(**params)
    elif 'cnn' in exp_method_conf['method']:
        if params is None:
            clf = build_model_v2(**exp_method_conf['params'])
        else:
            clf = build_model_v2(**params)
    else:
        sys.exit("unknown method: {}".format(exp_method_conf['method']))

    print_verbose("Method chosen: {}".format(clf))
    if 'dnn' in exp_method_conf['method'] or 'cnn' in exp_method_conf['method']:
        print_verbose("Method params: {}".format(clf.get_config()))
    else:
        print_verbose("Method params: {}".format(clf.get_params().keys()))
    return clf


def process_power_treces(traces,
                         fname):
    list_time = []
    list_tags = []
    list_durations = []
    list_energys = []
    list_gpu = []
    for trace in traces:
        for e in trace:
            list_time.append(e.timestamp)
            list_tags.append(e.tag)
            list_durations.append(e.duration)
            list_energys.append(e.energy['package_0'])
            list_gpu.append(e.energy['nvidia_gpu_0'])
    eng_rep = {
        'timestamp': list_time,
        'tag': list_tags,
        'duration': list_durations,
        'energy': list_energys,
        'gpu': list_gpu,
    }
    df_report = pd.DataFrame(eng_rep)
    df_report.to_csv(fname)

def cv_exp(conf,
           clf,
           sss,
           X,
           y,
           definitions,
           model_dir,
           exp_method_conf,
           images=None,
           nice_y=None,
           image_shape=None,
           ):
    '''
    Cross validation experiment
    :param conf:  configuration file
    :param clf: ML method instance
    :param sss: Stratified shuffle split instance
    :param X: input data (scaled)
    :param y: ground truth
    :param definitions:  class definitions
    :param model_dir: model directory
    :param exp_method_conf: ML method configuration
    :return:
    '''
    power_meter = conf['power_meter']
    # print("---->>>Power: {}".format(power))
    power_traces = []
    for i in range(conf['iterations']):
        print_verbose("Starting iteration {}".format(i))
        report = {
            "Accuracy": [],
            "BallancedAccuracy": [],
            "Jaccard": [],
            "StartTrainingTime": [],
            "EndTrainingTime": [],
            "DurationTraining": [],
            "StartTestingTime": [],
            "EndTestingTime": [],
            "DurationTesting": [],
        }
        fold = 1
        for train_index, test_index in sss.split(X, y):
            print_verbose("Starting fold {}".format(fold))
            if exp_method_conf["method"] == 'cnn':
                try:
                    train_dir, val_dir = generate_images(images, train_index, test_index, labels=nice_y)
                except Exception as inst:
                    print(f"Error while generating images for CNN with {type(inst)} and {inst.args}")
                    sys.exit()

                # hardcoded params
                batch_size = 32
                epochs = 1000
                patience = 10

                start_time_train = time.time()
                datagen = tf.keras.preprocessing.image.ImageDataGenerator()
                train_generator = datagen.flow_from_directory(
                    train_dir,
                    color_mode='grayscale',
                    shuffle=False,
                    target_size=image_shape,
                    batch_size=batch_size,
                    class_mode='categorical'
                )

                valid_generator = datagen.flow_from_directory(
                    val_dir,
                    color_mode='grayscale',
                    shuffle=False,
                    target_size=image_shape,
                    batch_size=batch_size,
                    class_mode='categorical'
                )

                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                                 patience=5, min_lr=0.001)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                  patience=patience)  # early stop patience
                if power_meter:
                    meter.start(tag=f'CV_{exp_method_conf["method"]}_Iteration_{i}_Fold_{fold}_train')
                history = clf.fit(train_generator,
                                    steps_per_epoch=train_generator.samples // batch_size,
                                    epochs=epochs,
                                    validation_data=valid_generator,
                                    validation_steps=valid_generator.samples // batch_size,
                                    verbose=1,
                                    callbacks=[early_stopping, reduce_lr]
                                    )
            else:
                Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
                ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

                print_verbose("Start training ....")
                start_time_train = time.time()
                if power_meter:
                    meter.start(tag=f'CV_{exp_method_conf["method"]}_Iteration_{i}_Fold_{fold}_train')
                if exp_method_conf["method"] == 'dnn':
                    patience = 10
                    batch_size = 32
                    epochs = 1000
                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                  patience=5, min_lr=0.001)
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=patience)  # early stop patience
                    y_oh = pd.get_dummies(ytrain, prefix='target')
                    history = clf.fit(np.asarray(Xtrain), np.asarray(y_oh),
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      callbacks=[early_stopping,
                                                 reduce_lr
                                                 ],
                                      validation_split=0.33,
                                      verbose=1)
                else:
                    if power_meter:
                        meter.start(tag=f'CV_{exp_method_conf["method"]}_Iteration_{i}_Fold_{fold}_train')
                    clf.fit(Xtrain, ytrain)
            end_time_train = time.time()
            train_time = end_time_train - start_time_train

            report['StartTrainingTime'].append(start_time_train)
            report['EndTrainingTime'].append(end_time_train)
            report['DurationTraining'].append(train_time)

            print_verbose("Predicting ....")
            start_time_test = time.time()
            if power_meter:
                meter.record(tag=f'CV_{exp_method_conf["method"]}_Iteration_{i}_Fold_{fold}_predict')
            if exp_method_conf["method"] == 'cnn':
                train_generator.reset()
                ypred = clf.predict_generator(train_generator,
                                              # steps=train_generator.samples // batch_size
                                              )
                ytest = train_generator.classes
            else:
                ypred = clf.predict(Xtest)
            if exp_method_conf["method"] == 'dnn' or exp_method_conf["method"] == 'cnn':
                ypred = np.argmax(ypred, axis=1)

            if power_meter:
                meter.stop()
                trace = meter.get_trace()
                power_traces.append(trace)
            end_time_test = time.time()
            predict_time = end_time_test - start_time_test

            report['StartTestingTime'].append(start_time_test)
            report['EndTestingTime'].append(end_time_test)
            report['DurationTesting'].append(predict_time)

            # Accuracy score
            acc = accuracy_score(ytest, ypred)
            report['Accuracy'].append(acc)
            print_verbose("Accuracy score fold {} is: {}".format(fold, acc))
            bacc = balanced_accuracy_score(ytest, ypred)
            report['BallancedAccuracy'].append(bacc)
            print_verbose("Ballanced accuracy fold {} score is: {}".format(fold, bacc))
            jaccard = jaccard_score(ytest, ypred, average='micro')
            print_verbose("Jaccard score fold {}: {}".format(fold, jaccard))
            report['Jaccard'].append(jaccard)

            # Classification report
            print_verbose("Full classification report for fold {}".format(fold))
            print_verbose(classification_report(ytest, ypred, digits=4, target_names=definitions))

            cf_report = classification_report(ytest, ypred, output_dict=True, digits=4, target_names=definitions)
            df_classification_report = pd.DataFrame(cf_report).transpose()
            print_verbose("Saving classification report")
            classification_rep_name = "{}_classification_{}_iteration_{}_fold_{}.csv".format(exp_method_conf['method'],
                                                                                             conf['experiment'], i,
                                                                                             fold)
            df_classification_report.to_csv(os.path.join(model_dir, classification_rep_name), index=False)

            print_verbose("Imbalanced Classification report for fold {}".format(fold))
            print_verbose(classification_report_imbalanced(ytest, ypred, digits=4, target_names=definitions))
            imb_cf_report = classification_report_imbalanced(ytest, ypred, output_dict=True, digits=4,
                                                             target_names=definitions)
            df_imb_classification_report = pd.DataFrame(imb_cf_report).transpose()
            print_verbose("Saving imbalanced classification report")
            imb_classification_rep_name = "{}_imb_classification_{}_iteration_{}_fold_{}.csv".format(
                exp_method_conf['method'],
                conf['experiment'], i, fold)
            df_imb_classification_report.to_csv(os.path.join(model_dir, imb_classification_rep_name), index=False)

            # Confusion matrix
            print_verbose("Generating confusion matrix fold {}".format(fold))
            cf_matrix = confusion_matrix(ytest, ypred)
            ht_cf = sns.heatmap(cf_matrix, annot=True, yticklabels=list(definitions), xticklabels=list(definitions))
            plt.title('Confusion Matrix Iteration {} Fold {}'.format(i, fold), fontsize=15)  # title with fontsize 20
            plt.xlabel('Ground Truth', fontsize=10)  # x-axis label with fontsize 15
            plt.ylabel('Predictions', fontsize=10)  # y-axis label with fontsize 15
            cf_fig = "CM_{}_{}_iteration_{}_fold_{}.png".format(exp_method_conf['method'], conf['experiment'], i, fold)
            ht_cf.figure.savefig(os.path.join(model_dir, cf_fig), bbox_inches='tight')
            plt.close()

            if exp_method_conf['method'] == 'dnn' or exp_method_conf['method'] == 'cnn':
                ml_meth_plot = exp_method_conf['method']
                print_verbose("Summerise training history ...")
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title(f'{ml_meth_plot}_iteration_{i}_fold_{fold} accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                h_acc_fig = "DNN_Acc_{}_{}_{}.png".format(ml_meth_plot, i, fold)
                plt.savefig(os.path.join(model_dir, h_acc_fig))


                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title(f'{ml_meth_plot}_iteration_{i}_fold_{fold} loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                h_loss_fig = "DNN_Loss_{}_{}_{}.png".format(ml_meth_plot, i, fold)
                plt.savefig(os.path.join(model_dir, h_loss_fig))

                # DNN history export
                df_history = pd.DataFrame(history.history)
                history_name = "{}_{}_{}_history.csv".format(ml_meth_plot, i, fold)
                df_history.to_csv(os.path.join(model_dir, history_name), index=False)

            else:
                # Feature importance
                print_verbose("Extracting Feature importance ...")
                feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
                sorted_feature = feat_importances.sort_values(ascending=True)
                # Limit number of sorted features
                sorted_feature = sorted_feature.tail(20)
                n_col = len(sorted_feature)

                # Plot the feature importances of the forest
                # plt.figure(figsize=(10,20), dpi=600) For publication only
                plt.title("Feature importances Fold {}".format(fold), fontsize=15)
                plt.barh(range(n_col), sorted_feature,
                         color="r", align="center")
                # If you want to define your own labels,
                # change indices to a list of labels on the following line.
                plt.yticks(range(n_col), sorted_feature.index)
                plt.ylim([-1, n_col])
                fi_fig = "FI_{}_{}_iteration_{}_fold_{}.png".format(exp_method_conf['method'], conf['experiment'], i, fold)
                plt.savefig(os.path.join(model_dir, fi_fig), bbox_inches='tight')

            # increment fold count
            fold += 1
        print_verbose("Saving final report ...")
        # Validation Report
        df_report = pd.DataFrame(report)
        final_report = "Model_{}_{}_iteration_{}_cv_report.csv".format(exp_method_conf['method'], conf['experiment'], i)
        df_report.to_csv(os.path.join(model_dir, final_report), index=False)
        if power_meter:
            process_power_treces(power_traces, os.path.join(model_dir, "Model_{}_{}_iteration_{}_cv_power_report.csv".format(exp_method_conf['method'], conf['experiment'], i)))


def learning_dataprop(dist,
                      clf,
                      X,
                      y,
                      model_dir,
                      conf,
                      exp_method_conf,
                      images=None,
                      image_shape=None,
                      ):
    """
    Plot learning curve for different data sample proportion
    :param dist: dataset proportion
    :param clf: ML model
    :param X: training data
    :param y: ground truth
    :param model_dir: model directory location
    :param conf: experiment cong
    :param exp_method_conf: experiment method conf
    :return:
    """
    power_meter = conf['power_meter']
    score_sets = []
    start_train = []
    end_train = []
    train_time = []
    start_predict = []
    end_predict = []
    predict_time = []
    data_frac = []
    power_traces = []
    for frac in dist:
        if exp_method_conf["method"] == 'cnn':
            tf.keras.backend.clear_session()
            X_subset = X.head(int(X.shape[0] * frac))
            y_subset = y.head(int(X.shape[0] * frac))
        else:
            X_subset = X.sample(frac=frac)
            y_subset = y.sample(frac=frac)
        data_frac.append(frac)

        if exp_method_conf["method"] == 'cnn':
            patience = 10
            batch_size = 32
            epochs = 1000
            clf_subsample_model = build_model_v2(**exp_method_conf['params'])
            sss_sub = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            for i, (train_index, test_index) in enumerate(sss_sub.split(X_subset, y_subset)):
                train_data_index = train_index
                test_data_index = test_index
            train_dir, val_dir = generate_images(images, train_data_index, test_data_index, labels=y_subset)
            datagen_sub = tf.keras.preprocessing.image.ImageDataGenerator()
            train_generator_sub = datagen_sub.flow_from_directory(
                train_dir,
                color_mode='grayscale',
                shuffle=False,
                target_size=image_shape,
                batch_size=batch_size,
                class_mode='categorical'
            )

            valid_generator_sub = datagen_sub.flow_from_directory(
                val_dir,
                color_mode='grayscale',
                shuffle=False,
                target_size=image_shape,
                batch_size=batch_size,
                class_mode='categorical'
            )
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                             patience=5, min_lr=0.001)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=patience)  # early stop patience
            start_time_train = time.time()
            if power_meter:
                meter.start(tag=f"{exp_method_conf['method']}_train_{frac}")
            history = clf_subsample_model.fit(train_generator_sub,
                                              steps_per_epoch=train_generator_sub.samples // batch_size,
                                              epochs=epochs,
                                              validation_data=valid_generator_sub,
                                              validation_steps=valid_generator_sub.samples // batch_size,
                                              verbose=1,
                                              callbacks=[early_stopping, reduce_lr]
                                              )
        else:
            if exp_method_conf["method"] == 'dnn':
                patience = 10
                batch_size = 32
                epochs = 1000
                start_time_train = time.time()
                if power_meter:
                    meter.start(tag=f"{exp_method_conf['method']}_train_{frac}")
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                                 patience=5, min_lr=0.001)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=patience)  # early stop patience
                y_oh = pd.get_dummies(y_subset, prefix='target')
                history = clf.fit(np.asarray(X_subset), np.asarray(y_oh),
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=[early_stopping,
                                             reduce_lr
                                             ],
                                  validation_split=0.33,
                                  verbose=1)
            else:
                if power_meter:
                    meter.start(tag=f"{exp_method_conf['method']}_train_{frac}")
                clf.fit(X_subset, y_subset)
        end_time_train = time.time()
        start_train.append(start_time_train)
        end_train.append(end_time_train)
        print_verbose("Training time: {}".format(end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)
        start_predict_time = time.time()
        if power_meter:
            meter.record(tag=f"{exp_method_conf['method']}_predict_{frac}")
        if exp_method_conf["method"] == 'cnn':
            train_generator_sub.reset()
            ypredict = clf_subsample_model.predict_generator(train_generator_sub,
                                                             # steps=train_generator.samples // batch_size
                                                             )
            y_subset = train_generator_sub.classes
        else:
            ypredict = clf.predict(X_subset)
        if exp_method_conf["method"] == 'dnn' or exp_method_conf["method"] == 'cnn':
            ypredict = np.argmax(ypredict, axis=1)

        if power_meter:
            meter.stop()
            power_traces.append(meter.get_trace())
        end_predict_time = time.time()
        start_predict.append(start_predict_time)
        end_predict.append(end_predict_time)
        print_verbose("Prediction time: {}".format(end_predict_time - start_predict_time))
        predict_time.append(end_predict_time - start_predict_time)
        f1_weighted_score = f1_score(y_subset, ypredict, average='weighted')
        score_sets.append(f1_weighted_score)

    report_dct = {
        'start_train': start_train,
        'end_train': end_train,
        'train_time': train_time,
        'predict_time': predict_time,
        'start_predict': start_predict,
        'end_predict': end_predict,
        'score': score_sets,
        'data_fraction': data_frac
    }
    df_report = pd.DataFrame(report_dct)
    df_report.to_csv(os.path.join(model_dir,
                                  f"{exp_method_conf['method']}_{conf['experiment']}_learningcurve_datafraction.csv"),
                     index=False)

    if power_meter:
        process_power_treces(power_traces,
                             os.path.join(model_dir, "Model_{}_{}_datafraction_power_report.csv".format(exp_method_conf['method'],
                                                                                   conf['experiment'])))
    # Plot learningcurve
    plt.figure(dpi=600)
    plt.grid()
    plt.plot(dist, score_sets, marker='o')
    plt.ylabel('F1')
    plt.xlabel('Data Sample Fraction')
    plt.savefig(os.path.join(model_dir, f"{exp_method_conf['method']}_{conf['experiment']}_learningcurve_datafraction.png"))



def rfe_ser(clf,
            sss,
            X,
            y,
            model_dir,
            exp_id,
            exp_method_conf,
            conf,
            fi=None,
            nice_y=None,):
    # if exp_method_conf["method"] == 'cnn':
    #     print("Currently not supported for CNN models")
    #     return 0
    df_iter = pd.DataFrame(index=X.index)
    if fi is None:
        fi = X.columns.values
    else:
        fi = fi
    power_meter = conf['power_meter']
    print_verbose("RFE Started, using {} ...".format(len(fi)))
    rfe_start_time = time.time()
    np_train_scores = np.empty((0, sss.n_splits))
    np_test_scores = np.empty((0, sss.n_splits))
    np_train_time = np.empty((0, sss.n_splits))
    np_predict_test_time = np.empty((0, sss.n_splits))
    np_predict_train_time = np.empty((0, sss.n_splits))
    feature_num = []
    power_traces = []
    for col in fi:
        df_iter[col] = X[col]
        start_train = []
        end_train = []
        train_time = []
        start_predict = []
        end_predict = []
        start_predict_train = []
        end_predict_train = []
        predict_time = []
        predict_time_train = []

        cv_scores_test = []
        cv_scores_train = []

        if exp_method_conf["method"] == 'cnn':
            model_rfe = Tab2Img()
            # print(df_iter.shape)
            # print(df_iter.shape[-1])
            if df_iter.shape[-1] < 40:  # skip if only one feature
                continue
            else:
                # print(f'+++{df_iter.shape}')
                images = model_rfe.fit_transform(np.asarray(df_iter), np.asarray(y))
                # print(images.shape)

        fold=1
        for train_index, test_index in sss.split(df_iter, y):
            if exp_method_conf["method"] == 'cnn':
                try:
                    # print(f'before gen: {images.shape}')
                    train_dir, val_dir = generate_images(images, train_index, test_index, labels=nice_y)
                    # print(f'after gen: {images.shape}')
                except Exception as inst:
                    print(f"Error while generating images for CNN with {type(inst)} and {inst.args}")
                    sys.exit()
            else:
                Xtrain, Xtest = df_iter.iloc[train_index], df_iter.iloc[test_index]
                ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

            if exp_method_conf["method"] == 'cnn':
                patience = 10
                batch_size = 32
                epochs = 1000
                datagen_rfe = tf.keras.preprocessing.image.ImageDataGenerator()
                train_generator_rfe = datagen_rfe.flow_from_directory(
                    train_dir,
                    color_mode='grayscale',
                    shuffle=False,
                    target_size=images.shape[-2:],
                    batch_size=batch_size,
                    class_mode='categorical'
                )

                valid_generator_rfe = datagen_rfe.flow_from_directory(
                    val_dir,
                    color_mode='grayscale',
                    shuffle=False,
                    target_size=images.shape[-2:],
                    batch_size=batch_size,
                    class_mode='categorical'
                )
                # print(f"after generator {images.shape}")
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                                 patience=5, min_lr=0.001)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                                  patience=patience)  # early stop patience
                start_time_train = time.time()
                if power_meter:
                    meter.start(tag=f"{exp_method_conf['method']}_train_{col}_fold_{fold}")
                # print(images.shape)
                # print(images.shape[-2])
                # print(images.shape[-1])
                clf = build_model_v2(**exp_method_conf['params'], input_c1=images.shape[-2], input_c2=images.shape[-1])
                history = clf.fit(train_generator_rfe,
                                                  steps_per_epoch=train_generator_rfe.samples // batch_size,
                                                  epochs=epochs,
                                                  validation_data=valid_generator_rfe,
                                                  validation_steps=valid_generator_rfe.samples // batch_size,
                                                  verbose=1,
                                                  callbacks=[early_stopping, reduce_lr]
                                                  )
            elif exp_method_conf["method"] == 'dnn':
                patience = 10
                batch_size = 32
                epochs = 1000
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                                 patience=5, min_lr=0.001)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                                  patience=patience)  # early stop patience
                y_oh = pd.get_dummies(ytrain, prefix='target')
                clf = dnn_serrano(**exp_method_conf['params'], input_dim=len(df_iter.columns.values))
                start_time_train = time.time()
                if power_meter:
                    meter.start(tag=f"{exp_method_conf['method']}_train_{col}_fold_{fold}")
                history = clf.fit(np.asarray(Xtrain), np.asarray(y_oh),
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=[early_stopping,
                                             reduce_lr
                                             ],
                                  validation_split=0.33,
                                  verbose=1)
            else:
                clf.fit(Xtrain, ytrain)
            end_time_train = time.time()
            start_train.append(start_time_train)
            end_train.append(end_time_train)
            print_verbose("Training time: {}".format(end_time_train - start_time_train))
            train_time.append(end_time_train - start_time_train)
            start_predict_time_train = time.time()
            if power_meter:
                meter.record(tag=f"{exp_method_conf['method']}_predict_train_{col}_fold_{fold}")
            if exp_method_conf["method"] == 'cnn':
                train_generator_rfe.reset()
                ypredict_train = clf.predict_generator(train_generator_rfe,
                                                       # steps=train_generator.samples // batch_size
                                                       )
                ytrain = train_generator_rfe.classes
            else:
                ypredict_train = clf.predict(Xtrain)
            if exp_method_conf["method"] == 'dnn' or exp_method_conf["method"] == 'cnn':
                ypredict_train = np.argmax(ypredict_train, axis=1)
            end_predict_time_train = time.time()
            start_predict_train.append(start_predict_time_train)
            end_predict_train.append(end_predict_time_train)
            print_verbose("Prediction time train : {}".format(end_predict_time_train - start_predict_time_train))
            predict_time_train.append(end_predict_time_train - start_predict_time_train)

            start_predict_time = time.time()
            if power_meter:
                meter.record(tag=f"{exp_method_conf['method']}_predict_test_{col}_fold_{fold}")
            if exp_method_conf["method"] == 'cnn':
                valid_generator_rfe.reset()
                ypredict_test = clf.predict_generator(valid_generator_rfe,
                                                      # steps=train_generator.samples // batch_size
                                                      )
                ytest = valid_generator_rfe.classes
            else:
                ypredict_test = clf.predict(Xtest)
            if exp_method_conf["method"] == 'dnn' or exp_method_conf["method"] == 'cnn':
                ypredict_test = np.argmax(ypredict_test, axis=1)

            if power_meter:
                meter.stop()
                trace = meter.get_trace()
                power_traces.append(trace)
            end_predict_time = time.time()
            print_verbose("Prediction time: {}".format(end_predict_time - start_predict_time))

            start_predict.append(start_predict_time)
            end_predict.append(end_predict_time)

            predict_time.append(end_predict_time - start_predict_time)
            f1_weighted_score_train = f1_score(ytrain, ypredict_train, average='weighted')
            f1_weighted_score_test = f1_score(ytest, ypredict_test, average='weighted')
            cv_scores_test.append(f1_weighted_score_test)
            cv_scores_train.append(f1_weighted_score_train)
        feature_num.append(len(df_iter.columns.values))
        np_train_scores = np.append(np_train_scores, [cv_scores_train], axis=0)
        np_test_scores = np.append(np_test_scores, [cv_scores_test], axis=0)
        np_train_time = np.append(np_train_time, [train_time], axis=0)
        np_predict_train_time = np.append(np_predict_train_time, [predict_time_train], axis=0)
        np_predict_test_time = np.append(np_predict_test_time, [predict_time], axis=0)

    rfe_stop_time = time.time()
    # Compute for fill_between plot
    np_train_scores_mean = np.mean(np_train_scores, axis=1)
    np_train_scores_std = np.std(np_train_scores, axis=1)
    np_test_scores_mean = np.mean(np_test_scores, axis=1)
    np_test_scores_std = np.std(np_test_scores, axis=1)

    # Aditional data
    np_train_time_mean = np.mean(np_train_time, axis=1)
    np_train_time_std = np.std(np_train_time, axis=1)
    np_predict_train_time_mean = np.mean(np_predict_train_time, axis=1)
    np_predict_train_time_std = np.std(np_predict_train_time, axis=1)
    np_predict_test_time_mean = np.mean(np_predict_test_time, axis=1)
    np_predict_test_time_std = np.std(np_predict_test_time, axis=1)

    report_times = {
        'train_time': np_train_time_mean,
        'train_time_std': np_train_time_std,
        'predict_train_time': np_predict_train_time_mean,
        'predict_train_time_std': np_predict_train_time_std,
        'predict_test_time': np_predict_test_time_mean,
        'predict_test_time_std': np_predict_test_time_std,
    }

    df_report_times = pd.DataFrame(report_times)
    df_report_times.to_csv(os.path.join(model_dir, f"{exp_method_conf['method']}_{exp_id}_cv_times_rfe.csv"), index=False)

    # Plot
    plt.figure(dpi=600)
    plt.grid()
    ft_num = [*range(1, len(feature_num)+1, 1)]

    plt.fill_between(ft_num, np_train_scores_mean - np_train_scores_std,
                     np_train_scores_mean + np_train_scores_std, alpha=0.1,
                     color="c")
    plt.fill_between(ft_num, np_test_scores_mean - np_test_scores_std,
                     np_test_scores_mean + np_test_scores_std, alpha=0.1, color="g")
    plt.plot(ft_num, np_train_scores_mean, 'o-', color="c",
             label="Training score")
    plt.plot(ft_num, np_test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # Labels and legends
    plt.xticks(ticks=ft_num, labels=ft_num, fontsize=14)
    plt.ylabel("F1 Score")
    plt.xlabel("Number of Features")
    # plt.legend(loc='upper right')
    plt.legend(loc="best")
    plt.savefig(os.path.join(model_dir, f"{exp_method_conf['method']}_{exp_id}_rfe.png"))

    report_rfe = {
        'rfe_start_time': rfe_start_time,
        'rfe_stop_time': rfe_stop_time,
        'rfe_total_time': rfe_stop_time - rfe_start_time,
    }

    df_rfe_report = pd.DataFrame(report_rfe, index=[0])
    df_rfe_report.to_csv(os.path.join(model_dir, f"{exp_method_conf['method']}_{exp_id}_times_rfe.csv"), index=False)

    if power_meter:
        process_power_treces(power_traces,
                             os.path.join(model_dir, "Model_{}_{}_rfe_power_report.csv".format(exp_method_conf['method'],
                                                                                conf['experiment'])))
        # report_dct ={
        #     'start_train': start_train,
        #     'end_train': end_train,
        #     'train_time': train_time,
        #     'predict_time': predict_time,
        #     'start_predict': start_predict,
        #     'end_predict': end_predict,
        #     'score': score_sets,
        #     'feature_num': feature_num
        # }

    # df_report = pd.DataFrame(report_dct)
    # df_report.to_csv(os.path.join(model_dir,
    #                               f"{exp_method_conf['method']}_{conf['experiment']}_rfe.csv"),
    #                  index=False)
    #
    # # Plot learningcurve
    # plt.figure(dpi=600)
    # plt.grid()
    # plt.plot(feature_num, score_sets, marker='o')
    # plt.ylabel('F1')
    # plt.xlabel('Features')
    # plt.savefig(
    #     os.path.join(model_dir, f"{exp_method_conf['method']}_{conf['experiment']}_rfe.png"))
    # plt.show()


def validation_curve(X,
                     y,
                     sss,
                     exp_method_conf,
                     model_dir,
                     exp_id,
                     conf,
                     images=None,
                     image_shape=None,
                     nice_y=None,
                        ):
    """
        :param X: Dataset for training
        :param y: Ground Truth of training data
        :param sss: cv instance
        :param exp_method_conf: method configuration for parameters
        :param model_dir: models directory
        :param cv: experiment unique id

        :return:
        """
    # Read original params
    method_name = exp_method_conf['method']

    power_meter = conf['power_meter']
    power_traces = []
    for i, clf_param in enumerate(exp_method_conf['validationcurve']):
        np_train_scores = np.empty((0, sss.n_splits))
        np_test_scores = np.empty((0, sss.n_splits))


        # Generate param range
        param_name = clf_param['param_name']
        print_verbose('Computing Validation Curve for {} Parameter name: {}'.format(method_name, param_name))
        param_range_values = clf_param['param_range']
        param_range = []
        for param in param_range_values:
            param_range.append({param_name: param})

        # Get param name and values
        param_values = []
        param_name = list(param_range[0].keys())[0]
        for d in param_range:
            param_values.append(list(d.values())[0])
        param_labels = param_values

        # Check if param values have string type
        if any(isinstance(item, str) for item in param_values):
            len_param_values = len(param_values)
            param_values = list(range(0, len_param_values))
        org_params = exp_method_conf['params']
        for param in param_range:
            params = org_params.copy()
            params.update(param)
            print_verbose(f"Starting params: {params}")
            cv_scores_train = []
            cv_scores_test = []
            start_train = []
            start_predict = []
            end_train = []
            end_predict = []
            train_time = []
            predict_time = []
            fold = 1
            for train_index, test_index in sss.split(X, y):
                if exp_method_conf["method"] == 'cnn':
                    patience = 10
                    batch_size = 32
                    epochs = 1000
                    tf.keras.backend.clear_session()
                    train_data_index = train_index
                    test_data_index = test_index

                    train_dir, val_dir = generate_images(images, train_data_index, test_data_index, labels=nice_y)
                    datagen_val = tf.keras.preprocessing.image.ImageDataGenerator()
                    train_generator_val = datagen_val.flow_from_directory(
                        train_dir,
                        color_mode='grayscale',
                        shuffle=False,
                        target_size=image_shape,
                        batch_size=batch_size,
                        class_mode='categorical'
                    )

                    valid_generator_val = datagen_val.flow_from_directory(
                        val_dir,
                        color_mode='grayscale',
                        shuffle=False,
                        target_size=image_shape,
                        batch_size=batch_size,
                        class_mode='categorical'
                    )
                    optimizer = params.pop('optimizer', tf.keras.optimizers.Adam(lr=0.0001))
                    print(f"Optimizer is set to: {optimizer}")
                    model_clf = build_model_v2(**params)
                    params.update({"optimizer": optimizer})
                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                                     patience=5, min_lr=0.001)
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                                      patience=patience)  # early stop patience
                    start_train_time = time.time()
                    if power_meter:
                        meter.start(tag=f"train_{method_name}_{param}_{fold}_{exp_id}")
                    history = model_clf.fit(train_generator_val,
                                            steps_per_epoch=train_generator_val.samples // batch_size,
                                            epochs=epochs,
                                            validation_data=valid_generator_val,
                                            validation_steps=valid_generator_val.samples // batch_size,
                                            verbose=1,
                                            callbacks=[early_stopping, reduce_lr]
                                            )
                else:
                    Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
                    ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
                    clf_param = select_method(exp_method_conf, params)
                    start_train_time = time.time()
                    if power_meter:
                        meter.start(tag=f"train_{method_name}_{param}_{fold}_{exp_id}")
                    if exp_method_conf["method"] == 'dnn':
                        patience = 10
                        batch_size = 32
                        epochs = 1000
                        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                                         patience=5, min_lr=0.001)
                        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                                          patience=patience)  # early stop patience
                        y_oh = pd.get_dummies(ytrain, prefix='target')
                        history = clf_param.fit(np.asarray(Xtrain), np.asarray(y_oh),
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          callbacks=[early_stopping,
                                                     reduce_lr
                                                     ],
                                          validation_split=0.33,
                                          verbose=1)
                    else:
                        start_train_time = time.time()
                        if power_meter:
                            meter.start(tag=f"train_{method_name}_{param}_{fold}_{exp_id}")
                        clf_param.fit(Xtrain, ytrain)
                end_train_time = time.time()

                print_verbose("Training time: {}".format(end_train_time - start_train_time))
                start_train.append(start_train_time)
                end_train.append(end_train_time)
                train_time.append(end_train_time - start_train_time)

                start_predict_time = time.time()
                if power_meter:
                    meter.record(tag=f"predict_{method_name}_{param_name}_{param}_{fold}_{exp_id}")

                if exp_method_conf["method"] == 'cnn':
                    train_generator_val.reset()
                    ypred = model_clf.predict_generator(train_generator_val,
                                                        # steps=train_generator.samples // batch_size
                                                        )
                    ytrain = train_generator_val.classes
                else:
                    ypred = clf_param.predict(Xtrain)
                if exp_method_conf["method"] == 'dnn' or exp_method_conf["method"] == 'cnn':
                    ypred = np.argmax(ypred, axis=1)
                if power_meter:
                    meter.stop()
                    trace = meter.get_trace()
                    power_traces.append(trace)

                end_train_time = time.time()

                print_verbose("Prediction time: {}".format(end_train_time - start_predict_time))
                start_predict.append(start_predict_time)
                end_predict.append(end_train_time)
                predict_time.append(end_train_time - start_predict_time)


                f1_weighted_score_train = f1_score(ytrain, ypred, average='weighted')

                if exp_method_conf["method"] == 'cnn':
                    valid_generator_val.reset()
                    ypredict_test = model_clf.predict_generator(valid_generator_val,
                                                                 # steps=train_generator.samples // batch_size
                                                                 )
                    ytest = valid_generator_val.classes
                else:
                    ypredict_test = clf_param.predict(Xtest)
                if exp_method_conf["method"] == 'dnn' or exp_method_conf["method"] == 'cnn':
                    ypredict_test = np.argmax(ypredict_test, axis=1)
                f1_weighted_score_test = f1_score(ytest, ypredict_test, average='weighted')

                cv_scores_test.append(f1_weighted_score_test)
                cv_scores_train.append(f1_weighted_score_train)
                fold += 1
            report = {
                'start_train': start_train,
                'end_train': end_train,
                'train_time': train_time,
                'start_predict': start_predict,
                'end_predict': end_predict,
                'predict_time': predict_time,
                'f1_wighted_train': cv_scores_train,
                'f1_wighted_test': cv_scores_test,
            }
            df_report = pd.DataFrame(report)
            df_report.to_csv(os.path.join(model_dir,
                                          f"{method_name}_{exp_id}_validationcurve_{param_name}_{param}.csv"),
                             index=False)
            np_train_scores = np.append(np_train_scores, [cv_scores_train], axis=0)
            np_test_scores = np.append(np_test_scores, [cv_scores_test], axis=0)

        # Compute for fill_between plot
        np_train_scores_mean = np.mean(np_train_scores, axis=1)
        np_train_scores_std = np.std(np_train_scores, axis=1)
        np_test_scores_mean = np.mean(np_test_scores, axis=1)
        np_test_scores_std = np.std(np_test_scores, axis=1)

        # Plot
        plt.figure(dpi=600)
        plt.grid()
        plt.fill_between(param_values, np_train_scores_mean - np_train_scores_std,
                         np_train_scores_mean + np_train_scores_std, alpha=0.1,
                         color="c")
        plt.fill_between(param_values, np_test_scores_mean - np_test_scores_std,
                         np_test_scores_mean + np_test_scores_std, alpha=0.1, color="g")
        plt.plot(param_values, np_train_scores_mean, 'o-', color="c",
                 label="Training score")
        plt.plot(param_values, np_test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        # Labels and legends
        plt.xticks(ticks=param_values, labels=param_labels)
        plt.ylabel("F1 Score")
        plt.xlabel(param_name)
        # plt.legend(loc='upper right')
        plt.legend(loc="best")
        plt.savefig(os.path.join(model_dir, f"{method_name}_{exp_id}_validationcurve_{param_name}.png"))

        if power_meter:
            process_power_treces(power_traces,
                                 os.path.join(model_dir,
                                              "Model_{}_{}_validation_curves_power_report.csv".format(exp_method_conf['method'],
                                                                                        conf['experiment'])))
def run(conf):
    exp_dir, model_dir = check_data_folders(conf)
    # Load data
    df_anomaly, df_audsome, df_clean, df_clean_audsome = load_data()

    # set verbose mode
    # os.environ['SER_VERBOSE'] = int(conf['verbose'])

    if 'anomaly' == conf["dataset"]:
        data = df_anomaly
    elif 'audsome' == conf["dataset"]:
        data = df_audsome
    elif 'clean' == conf["dataset"]:
        data = df_clean
    elif 'clean_audsome' == conf["dataset"]:
        data = df_clean_audsome
    else:
        sys.exit("unknown dataset: {}".format(conf["dataset"]))
    print_verbose("Dataset chosen ...")
    # Nice print
    nice_y = data['target']

    # Uncomment for removing dummy
    # print("Removed Dummy class")
    # data.loc[data.target == "dummy", 'target'] = "0"

    # Creating the dependent variable class
    factor = pd.factorize(data['target'])
    data.target = factor[0]
    definitions = factor[1]

    # Splitting the data into independent and dependent variables
    X = data.drop('target', axis=1)
    y = data['target']


    with open(conf['file'], 'r') as cfile:
        if not os.path.isfile(conf['file']):
            print("Experimental conf file missing !!!")
            sys.exit(1)
        exp_method_conf = yaml.load(cfile, Loader=yaml.UnsafeLoader)

    # Scaling the data
    if exp_method_conf['method'] == 'cnn':
        # X = X.astype('float32')
        # X /= 255
        scaler = MinMaxScaler(feature_range=(0, 255), clip=True)
    else:
        scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)  #

    # Method selection
    # conf_file = {
    #     "method": "AdaBoostClassifier",
    #     "params": {
    #         "n_estimators": 200,
    #         "learning_rate": 0.1,
    #         "algorithm": "SAMME",
    #         "random_state": 42},

    #     "learningcurve":  np.linspace(0.3, 1.0, 10),
    #     "validationcurve": [
    #         {
    #             "param_name": "n_estimators",
    #             "param_range": [10, 60, 100, 200, 600, 1000, 2000]
    #         },
    #         {
    #             "param_name": "learning_rate",
    #             "param_range": [0.1, 0.3, 0.5, 0.7, 1.0]
    #         },
    #         {
    #             "param_name": "algorithm",
    #             "param_range": ["SAMME", "SAMME.R"]
    #         }
    #     ],
    #     "rfe": True,
    #
    # }

    # exp_method_conf = yaml.load(conf['file'], Loader=yaml.UnsafeLoader)
    print_verbose(exp_method_conf.keys())
    clf = select_method(exp_method_conf)

    #Fix lightgbm.basic.LightGBMError: Do not support special JSON characters in feature name.

    if 'LightGBM' in exp_method_conf['method']:
        import re
        X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    # if 'AdaBoostClassifier' in exp_method_conf['method']:
    #     clf = AdaBoostClassifier(**exp_method_conf['params'])
    #     print_verbose("Method chosen: {}".format(clf))
    #     print_verbose("Method params: {}".format(exp_method_conf['params']))
    # else:
    #     sys.exit("unknown method: {}".format(exp_method_conf['method']))
    #
    # print_verbose("Method chosen: {}".format(clf))
    # print_verbose("Method params: {}".format(clf.get_params().keys()))

    # Tabular to image transformation
    images = None
    IMAGE_SHAPE = None
    if 'cnn' in exp_method_conf['method']:
        model_tab = Tab2Img()
        images = model_tab.fit_transform(np.asarray(X), np.asarray(y))
        IMAGE_SHAPE = (8, 8)

    # Cross validation
    sss = StratifiedShuffleSplit(n_splits=conf['cross_validation'], test_size=conf['cross_validation_test'],
                                 random_state=42)

    # Experiment cv
    cv_exp(conf=conf,
           clf=clf,
           sss=sss,
           X=X,
           y=y,
           definitions=definitions,
           model_dir=model_dir,
           exp_method_conf=exp_method_conf,
           images=images,
           nice_y=nice_y,
           image_shape=IMAGE_SHAPE,
           )

    if 'datafraction' in exp_method_conf.keys():
        print_verbose(exp_method_conf['datafraction'])
        dist = np.linspace(**exp_method_conf['datafraction'])
        learning_dataprop(dist,
                          clf,
                          X,
                          y,
                          model_dir,
                          conf,
                          exp_method_conf,
                          images=images,
                          image_shape=IMAGE_SHAPE,
                          )

    if 'validationcurve' in exp_method_conf.keys():
        print_verbose(exp_method_conf['validationcurve'])
        validation_curve(X,
                         y,
                         sss,
                         exp_method_conf,
                         model_dir,
                         conf['experiment'],
                         conf,
                         images=images,
                         image_shape=IMAGE_SHAPE,
                         nice_y=nice_y)

    if 'rfe' in exp_method_conf.keys():
        print_verbose(exp_method_conf['rfe'])
        if isinstance(exp_method_conf['rfe'], list):
            rfe_ser(clf, X, y, model_dir, conf['experiment'], exp_method_conf, conf=conf, fi=exp_method_conf['rfe'], nice_y=nice_y)
        else:
            rfe_ser(clf, sss, X, y, model_dir, conf['experiment'], exp_method_conf, conf, nice_y=nice_y)

    print_verbose("Execution finished ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serrano Energy consumption experiments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", default='ada.yaml', type=str, help="configuration file")
    parser.add_argument("-d", "--dataset", type=str, default='anomaly', help="choose dataset, can be one of: anomaly, "
                                                                             "audsome, clean, clean_audsome")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="number of iterations")
    parser.add_argument("-cv", "--cross_validation", type=int, default=5, help="number of splits used for cv")
    parser.add_argument("-cvt", "--cross_validation_test", type=float, default=0.2, help="percent of test set")
    parser.add_argument("-e", "--experiment", type=str, default='exp_power', help="experiment unique identifier")
    parser.add_argument("-p", "--power_meter", type=int, default=1, help="power consumption")
    parser.add_argument("-v", "--verbose", action='store_true', help="verbosity")

    args = parser.parse_args()
    config = vars(args)
    # print("Configuration:")
    # for (k, v) in config.items():
    #     print("{}: {}".format(k, v))
    run(config)