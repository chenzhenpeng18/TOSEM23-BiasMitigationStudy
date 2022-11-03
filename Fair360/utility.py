from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset,MEPSDataset19
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
import numpy as np

def get_data(dataset_used, protected,preprocessed = False):
    if dataset_used == "adult":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_adult(['sex'])
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = load_preproc_data_adult(['race'])
        optim_options = {
            "distortion_fun": get_distortion_adult,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
        if not preprocessed:
            dataset_orig = AdultDataset().convert_to_dataframe()[0]
            dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")
    elif dataset_used == "german":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_german(['sex'])
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
            dataset_orig = load_preproc_data_german(['age'])
            optim_options = {
                "distortion_fun": get_distortion_german,
                "epsilon": 0.1,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
        if not preprocessed:
            dataset_orig = GermanDataset().convert_to_dataframe()[0]
            dataset_orig['credit'] = np.where(dataset_orig['credit'] == 1, 1, 0)
            dataset_orig.columns = dataset_orig.columns.str.replace("credit", "Probability")
    elif dataset_used == "compas":
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
            dataset_orig = load_preproc_data_compas(['sex'])
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
            dataset_orig = load_preproc_data_compas(['race'])
        optim_options = {
            "distortion_fun": get_distortion_compas,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }
        if not preprocessed:
            dataset_orig = CompasDataset().convert_to_dataframe()[0]
            dataset_orig.columns = dataset_orig.columns.str.replace("two_year_recid", "Probability")
            dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 1, 0, 1)
    elif dataset_used == "bank":
        privileged_groups = [{'age': 1}]  
        unprivileged_groups = [{'age': 0}]
        dataset_orig = BankDataset().convert_to_dataframe()[0]
        dataset_orig.rename(columns={'y': 'Probability'}, inplace=True)
        optim_options = None
    elif dataset_used == "mep":
        privileged_groups = [{'RACE': 1}]
        unprivileged_groups = [{'RACE': 0}]
        dataset_orig = MEPSDataset19().convert_to_dataframe()[0]
        dataset_orig.rename(columns={'UTILIZATION': 'Probability'}, inplace=True)
        optim_options = None
    return dataset_orig, privileged_groups,unprivileged_groups,optim_options


def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    return clf
