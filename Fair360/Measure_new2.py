from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,roc_auc_score,matthews_corrcoef
from aif360.metrics import ClassificationMetric

def measure_final_score(dataset_orig_test, dataset_orig_predict,privileged_groups,unprivileged_groups):
    poslabel = dataset_orig_test.favorable_label
    neglabel = dataset_orig_test.unfavorable_label

    y_test = dataset_orig_test.labels
    y_pred = dataset_orig_predict.labels

    accuracy = accuracy_score(y_test, y_pred)
    recall1 = recall_score(y_test, y_pred, pos_label=poslabel)
    recall0 = recall_score(y_test, y_pred, pos_label=neglabel)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision1 = precision_score(y_test, y_pred, pos_label=poslabel)
    precision0 = precision_score(y_test, y_pred, pos_label=neglabel)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score1 = f1_score(y_test, y_pred, pos_label=poslabel)
    f1score0 = f1_score(y_test, y_pred, pos_label=neglabel)
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    classified_metric_pred = ClassificationMetric(dataset_orig_test, dataset_orig_predict,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)

    spd = abs(classified_metric_pred.statistical_parity_difference())
    aod = classified_metric_pred.average_abs_odds_difference()
    eod = abs(classified_metric_pred.equal_opportunity_difference())
    erd = abs(classified_metric_pred.error_rate_difference())

    return accuracy, recall1, recall0, recall_macro, precision1, precision0, precision_macro, f1score1, f1score0, f1score_macro, mcc, spd,aod,eod,erd
