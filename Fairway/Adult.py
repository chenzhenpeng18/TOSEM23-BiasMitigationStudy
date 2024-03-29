# coding: utf-8
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from aif360.datasets import AdultDataset
from sklearn.preprocessing import MinMaxScaler
import argparse
import sys
sys.path.append(os.path.abspath('.'))

from Measure_new import measure_final_score
from remove_bias import remove_bias
from flash import flash_fair

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['lr'], help="Classifier name")
    parser.add_argument("-p", "--protected", type=str, required=True,
                        help="Protected attribute")

    args = parser.parse_args()
    model_type = args.clf
    protected_attribute = args.protected

    #Load dataset
    dataset_orig= AdultDataset().convert_to_dataframe()[0]
    dataset_orig.columns = dataset_orig.columns.str.replace("income-per-year", "Probability")

    val_name = "fairway_{}_adult_{}.txt".format(model_type, protected_attribute)
    fout = open(val_name, 'w')

    results = {}
    performance_index = ['accuracy', 'recall1', 'recall0', 'recall_macro', 'precision1', 'precision0', 'precision_macro', 'f1score1', 'f1score0', 'f1score_macro','mcc', 'spd', 'aod', 'eod','erd']
    for p_index in performance_index:
        results[p_index] = []

    repeat_time = 50
    for round_num in range(repeat_time):
        print(round_num)

        np.random.seed(round_num)
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle = True)

        #scale data
        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train=pd.DataFrame(scaler.transform(dataset_orig_train),columns = dataset_orig.columns)
        dataset_orig_test=pd.DataFrame(scaler.transform(dataset_orig_test),columns = dataset_orig.columns)

        print("training_size_original:", dataset_orig_train.shape)
        print("test_size_original:", dataset_orig_test.shape)

        #debias training data
        dataset_orig_train = remove_bias(dataset_orig_train, protected_attribute, model_type)

        print("training_size_after:", dataset_orig_train.shape)
        print("test_size_after:", dataset_orig_test.shape)


        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        #multiobjective optimization
        clf = flash_fair(dataset_orig_train, protected_attribute,model_type)

        print("-------------------------:Fairway")
        round_result = measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute)
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index+'\t')
        for i in range(repeat_time):
            fout.write('%f\t' % results[p_index][i])
        fout.write('\n')
    fout.close()
