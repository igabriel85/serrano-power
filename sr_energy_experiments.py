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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from subprocess import check_output
# from sklearn.externals import joblib
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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

    else:
        sys.exit("unknown method: {}".format(exp_method_conf['method']))

    print_verbose("Method chosen: {}".format(clf))
    print_verbose("Method params: {}".format(clf.get_params().keys()))
    return clf


def cv_exp(conf,
           clf,
           sss,
           X,
           y,
           definitions,
           model_dir,
           exp_method_conf
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
            Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

            print_verbose("Start training ....")
            start_time_train = time.time()
            clf.fit(Xtrain, ytrain)
            end_time_train = time.time()
            train_time = end_time_train - start_time_train

            report['StartTrainingTime'].append(start_time_train)
            report['EndTrainingTime'].append(end_time_train)
            report['DurationTraining'].append(train_time)

            print_verbose("Predicting ....")
            start_time_test = time.time()
            ypred = clf.predict(Xtest)
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

            # Feature importance
            print_verbose("Extracting Feature improtance ...")
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
        final_report = "Model_{}_{}_iteration_{}_report.csv".format(exp_method_conf['method'], conf['experiment'], i)
        df_report.to_csv(os.path.join(model_dir, final_report), index=False)


def learning_dataprop(dist,
                      clf,
                      X,
                      y,
                      model_dir,
                      conf,
                      exp_method_conf
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
    score_sets = []
    start_train = []
    end_train = []
    train_time = []
    start_predict = []
    end_predict = []
    predict_time = []
    data_frac = []
    for frac in dist:
        X_subset = X.sample(frac=frac)
        y_subset = y.sample(frac=frac)
        data_frac.append(frac)
        start_time_train = time.time()
        clf.fit(X_subset, y_subset)
        end_time_train = time.time()
        start_train.append(start_time_train)
        end_train.append(end_time_train)
        print_verbose("Training time: {}".format(end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)
        start_predict_time = time.time()
        ypredict = clf.predict(X_subset)
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


    # Plot learningcurve
    plt.figure(dpi=600)
    plt.grid()
    plt.plot(dist, score_sets, marker='o')
    plt.ylabel('F1')
    plt.xlabel('Data Sample Fraction')
    plt.savefig(os.path.join(model_dir, f"{exp_method_conf['method']}_{conf['experiment']}_learningcurve_datafraction.png"))
    plt.show()


def rfe_ser(clf,
            sss,
            X,
            y,
            model_dir,
            exp_id,
            exp_method_conf,
            fi=None):
    df_iter = pd.DataFrame(index=X.index)
    if fi is None:
        fi = X.columns.values
    else:
        fi = fi

    print_verbose("RFE Started, using {} ...".format(len(fi)))
    rfe_start_time = time.time()
    np_train_scores = np.empty((0, sss.n_splits))
    np_test_scores = np.empty((0, sss.n_splits))
    np_train_time = np.empty((0, sss.n_splits))
    np_predict_test_time = np.empty((0, sss.n_splits))
    np_predict_train_time = np.empty((0, sss.n_splits))
    feature_num = []
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
        for train_index, test_index in sss.split(df_iter, y):
            Xtrain, Xtest = df_iter.iloc[train_index], df_iter.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
            start_time_train = time.time()
            clf.fit(Xtrain, ytrain)
            end_time_train = time.time()
            start_train.append(start_time_train)
            end_train.append(end_time_train)
            print_verbose("Training time: {}".format(end_time_train - start_time_train))
            train_time.append(end_time_train - start_time_train)
            start_predict_time_train = time.time()
            ypredict_train = clf.predict(Xtrain)
            end_predict_time_train = time.time()
            start_predict_train.append(start_predict_time_train)
            end_predict_train.append(end_predict_time_train)
            print_verbose("Prediction time train : {}".format(end_predict_time_train - start_predict_time_train))
            predict_time_train.append(end_predict_time_train - start_predict_time_train)

            start_predict_time = time.time()
            ypredict_test = clf.predict(Xtest)
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
                     exp_id):
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
                Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
                ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
                clf_param = select_method(exp_method_conf, params)
                start_train_time = time.time()
                clf_param.fit(Xtrain, ytrain)
                end_train_time = time.time()

                print_verbose("Training time: {}".format(end_train_time - start_train_time))
                start_train.append(start_train_time)
                end_train.append(end_train_time)
                train_time.append(end_train_time - start_train_time)

                start_predict_time = time.time()
                ypred = clf_param.predict(Xtrain)
                end_train_time = time.time()

                print_verbose("Prediction time: {}".format(end_train_time - start_predict_time))
                start_predict.append(start_predict_time)
                end_predict.append(end_train_time)
                predict_time.append(end_train_time - start_predict_time)


                f1_weighted_score_train = f1_score(ytrain, ypred, average='weighted')

                ypredict_test = clf_param.predict(Xtest)
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

    # Scaling the data
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
    with open(conf['file'], 'r') as cfile:
        if not os.path.isfile(conf['file']):
            print("Experimental conf file missing !!!")
            sys.exit(1)
        exp_method_conf = yaml.load(cfile, Loader=yaml.UnsafeLoader)
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
           exp_method_conf=exp_method_conf)

    if 'datafraction' in exp_method_conf.keys():
        print_verbose(exp_method_conf['datafraction'])
        dist = np.linspace(**exp_method_conf['datafraction'])
        learning_dataprop(dist,
                          clf,
                          X,
                          y,
                          model_dir,
                          conf,
                          exp_method_conf
                          )

    if 'validationcurve' in exp_method_conf.keys():
        print_verbose(exp_method_conf['validationcurve'])
        validation_curve(X,
                         y,
                         sss,
                         exp_method_conf,
                         model_dir,
                         conf['experiment'])

    if 'rfe' in exp_method_conf.keys():
        print_verbose(exp_method_conf['rfe'])
        if isinstance(exp_method_conf['rfe'], list):
            rfe_ser(clf, X, y, model_dir, conf['experiment'], exp_method_conf, fi=exp_method_conf['rfe'])
        else:
            rfe_ser(clf, sss, X, y, model_dir, conf['experiment'], exp_method_conf)

    print_verbose("Execution finished ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serrano Energy consumption experiments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", default='cat.yaml', type=str, help="configuration file")
    parser.add_argument("-d", "--dataset", type=str, default='anomaly', help="choose dataset, can be one of: anomaly, "
                                                                             "audsome, clean, clean_audsome")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="number of iterations")
    parser.add_argument("-cv", "--cross_validation", type=int, default=5, help="number of splits used for cv")
    parser.add_argument("-cvt", "--cross_validation_test", type=float, default=0.2, help="percent of test set")
    parser.add_argument("-e", "--experiment", type=str, default='exp_6', help="experiment unique identifier")
    parser.add_argument("-v", "--verbose", action='store_true', help="verbosity")

    args = parser.parse_args()
    config = vars(args)
    # print("Configuration:")
    # for (k, v) in config.items():
    #     print("{}: {}".format(k, v))
    run(config)