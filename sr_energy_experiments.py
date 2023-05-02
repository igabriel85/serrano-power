"""
Serrano Energy consumption experiments
Goals:
- Run predictive models and measure energy consumption
    - Run predictive models and measure energy consumption with different
    - Run predicitive models on different datasets and measure energy consumption
"""

import argparse
import yaml
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score, make_scorer, classification_report, accuracy_score, jaccard_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from subprocess import check_output
# from sklearn.externals import joblib
from joblib import dump, load
import os
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
    print("Accuracy score is: {}".format(accuracy_score(y, y_pred)))
    print("Ballanced accuracy score is: {}".format(balanced_accuracy_score(y, y_pred)))
    print("Jaccard score (micro): {}".format(jaccard_score(y, y_pred, average='micro')))
    print("Jaccard score (macro): {}".format(jaccard_score(y, y_pred, average='macro')))
    print("Jaccard score (weighted): {}".format(jaccard_score(y, y_pred, average='weighted')))


    print("Full classification report")
    print(classification_report(y, y_pred, target_names=definitions))
    report = classification_report(y, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    classification_rep_name = "{}_classification_rep_best.csv".format(prefix)
    df_classification_report.to_csv(os.path.join(model_dir,classification_rep_name), index=False)


    print("Imbalanced Classification report")
    print(classification_report_imbalanced(y, y_pred, target_names=definitions))
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
    print("Dataset chosen ...")
    data = df_anomaly
    # drop_col = ['t1','t2','t3','t4']
    print("Remove unwanted columns ...")
    # print("Shape before drop: {}".format(data.shape))
    # data.drop(drop_col, axis=1, inplace=True)
    # print("Shape after drop: {}".format(data.shape))
    return df_anomaly, df_audsome, df_clean, df_clean_audsome


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
    if 'AdaBoostClassifier' in exp_method_conf['method']:
        clf = AdaBoostClassifier(**exp_method_conf['params'])
        print_verbose("Method chosen: {}".format(clf))
        print_verbose("Method params: {}".format(exp_method_conf['params']))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serrano Energy consumption experiments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file", default='ada.yaml', type=str, help="configuration file")
    parser.add_argument("-d", "--dataset", type=str, default='anomaly', help="choose dataset, can be one of: anomaly, "
                                                                             "audsome, clean, clean_audsome")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="number of iterations")
    parser.add_argument("-cv", "--cross_validation", type=int, default=5, help="number of splits used for cv")
    parser.add_argument("-cvt", "--cross_validation_test", type=float, default=0.2, help="percent of test set")
    parser.add_argument("-e", "--experiment", type=str, default='exp_1', help="experiment unique identifier")
    parser.add_argument("-v", "--verbose", action='store_true', help="verbosity")

    args = parser.parse_args()
    config = vars(args)
    print("Configuration:")
    for (k, v) in config.items():
        print("{}: {}".format(k, v))
    print(config)
    run(config)