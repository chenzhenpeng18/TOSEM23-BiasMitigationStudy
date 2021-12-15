import sys

sys.path.append("../")
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from numpy import mean,std
import copy
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utility2 import get_data, get_classifier
from Measure_new2 import measure_final_score
import os
import argparse

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="adult",
                    help="Dataset name")
parser.add_argument("-c", "--clf", type=str, default="lr",
                    help="Classifier name")
parser.add_argument("-p", "--protected", type=str, default="sex",
                    help="Protected attribute")

args = parser.parse_args()

dataset_used = args.dataset  # "adult", "german", "compas"
attr = args.protected
clf_name = args.clf

val_name = "op_{}_{}_{}.txt".format(clf_name,dataset_used,attr)

fout = open(val_name, 'w')

dataset_orig, privileged_groups, unprivileged_groups, optim_options = get_data(dataset_used, attr, preprocessed=True)

results = {}
performance_index = ['accuracy', 'recall1', 'recall0', 'recall_macro', 'precision1', 'precision0', 'precision_macro', 'f1score1', 'f1score0', 'f1score_macro', 'roc_auc', 'mcc', 'spd', 'di', 'aod', 'eod','erd']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 50
for r in range(repeat_time):
    print(r)
    np.random.seed(r)

    OP = OptimPreproc(OptTools, optim_options,
                      unprivileged_groups=unprivileged_groups,
                      privileged_groups=privileged_groups)

    # split training data and test data
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    OP = OP.fit(dataset_orig_train)

    dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)
    dataset_transf_train = dataset_orig_train.align_datasets(dataset_transf_train)

    clf = get_classifier(clf_name)
    clf = clf.fit(dataset_transf_train.features, dataset_transf_train.labels)

    pred = clf.predict(dataset_orig_test.features).reshape(-1, 1)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    dataset_orig_test_pred.labels = pred

    round_result = measure_final_score(dataset_used, dataset_orig_test, dataset_orig_test_pred, privileged_groups,
                                       unprivileged_groups)

    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index + '\t')
    for i in range(repeat_time):
        fout.write('%f\t' % results[p_index][i])
    fout.write('%f\t%f\n' % (mean(results[p_index]), std(results[p_index])))
fout.close()