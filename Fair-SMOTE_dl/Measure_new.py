import copy
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,roc_auc_score,matthews_corrcoef
from aif360.metrics import ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from tensorflow import keras

def get_classifier(name,datasize):
    if name =="dl":
        clf = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=datasize),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        clf.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
    elif name == "dl2":
        clf = keras.Sequential([
            keras.layers.Dense(50, activation="relu", input_shape=datasize),
            keras.layers.Dense(30, activation="relu"),
            keras.layers.Dense(15, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(5, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        clf.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
    elif name == "dl3":
        clf = keras.Sequential([
            keras.layers.Dense(30, activation="relu", input_shape=datasize),
            keras.layers.Dense(20, activation="relu"),
            keras.layers.Dense(15, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(5, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        clf.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
    elif name == "dl4":
        clf = keras.Sequential([
            keras.layers.Dense(30, activation="relu", input_shape=datasize),
            keras.layers.Dense(20, activation="relu"),
            keras.layers.Dense(15, activation="relu"),
            keras.layers.Dense(15, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        clf.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
    return clf

def measure_final_score(test_df, clf, X_train, y_train, X_test, y_test, biased_col):
    clf.fit(X_train, y_train,epochs=20)
    y_pred = clf.predict_classes(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall1 = recall_score(y_test, y_pred, pos_label=1)
    recall0 = recall_score(y_test, y_pred, pos_label=0)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision1 = precision_score(y_test, y_pred, pos_label=1)
    precision0 = precision_score(y_test, y_pred, pos_label=0)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score1 = f1_score(y_test, y_pred, pos_label=1)
    f1score0 = f1_score(y_test, y_pred, pos_label=0)
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    test_df_copy = copy.deepcopy(test_df)
    test_df_copy['Probability'] = y_pred

    tt1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=test_df, label_names=['Probability'],
                             protected_attribute_names=[biased_col])
    tt2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=test_df_copy, label_names=['Probability'],
                             protected_attribute_names=[biased_col])

    classified_metric_pred = ClassificationMetric(tt1, tt2, unprivileged_groups=[{biased_col: 0}], privileged_groups=[{biased_col: 1}])
    spd = abs(classified_metric_pred.statistical_parity_difference())
    aod = classified_metric_pred.average_abs_odds_difference()
    eod = abs(classified_metric_pred.equal_opportunity_difference())
    erd = abs(classified_metric_pred.error_rate_difference())

    return accuracy, recall1, recall0, recall_macro, precision1, precision0, precision_macro, f1score1, f1score0, f1score_macro, mcc, spd,aod,eod,erd
