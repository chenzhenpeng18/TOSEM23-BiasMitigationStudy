import numpy as np
import copy, math
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,roc_auc_score,matthews_corrcoef
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset

def measure_final_score(dataset_orig_test, dataset_orig_predict,privileged_groups,unprivileged_groups):

    y_test = dataset_orig_test.labels
    y_pred = dataset_orig_predict.labels

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy:", accuracy)
    recall1 = recall_score(y_test, y_pred, pos_label=1)
    print("recall@1:", recall1)
    recall0 = recall_score(y_test, y_pred, pos_label=0)
    print("recall@0:", recall0)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    print("recall@macro:", recall_macro)
    precision1 = precision_score(y_test, y_pred, pos_label=1)
    print("precision@1:", precision1)
    precision0 = precision_score(y_test, y_pred, pos_label=0)
    print("precision@0:", precision0)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    print("precision@macro:", precision_macro)
    f1score1 = f1_score(y_test, y_pred, pos_label=1)
    print("f1_score@1:", f1score1)
    f1score0 = f1_score(y_test, y_pred, pos_label=0)
    print("f1_score@0:", f1score0)
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    print("f1_score@macro:", f1score_macro)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("roc_auc_score:", roc_auc)
    mcc = matthews_corrcoef(y_test, y_pred)
    print("mcc_score:", mcc)

    classified_metric_pred = ClassificationMetric(dataset_orig_test, dataset_orig_predict,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)

    spd = abs(classified_metric_pred.statistical_parity_difference())
    print("SPD:", spd)
    di = abs(classified_metric_pred.disparate_impact())
    print("DI:", di)
    aod = classified_metric_pred.average_abs_odds_difference()
    print("AOD:", aod)
    eod = abs(classified_metric_pred.equal_opportunity_difference())
    print("EOD:", eod)
    erd = abs(classified_metric_pred.error_rate_difference())
    print("ERD:", erd)

    return accuracy, recall1, recall0, recall_macro, precision1, precision0, precision_macro, f1score1, f1score0, f1score_macro, roc_auc, mcc, spd,di,aod,eod,erd
