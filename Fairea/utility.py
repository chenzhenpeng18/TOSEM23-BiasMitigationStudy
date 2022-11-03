from aif360.datasets import AdultDataset, GermanDataset, CompasDataset, BankDataset,MEPSDataset19

def get_data(dataset_used, protected):
    """ Obtains dataset from AIF360.

    Parameters:
        dataset_used (str) -- Name of the dataset
        protected (str)    -- Protected attribute used
    Returns:
        dataset_orig (dataset)     -- Classifier with default configuration from scipy
        dataset_orig (dataset)     -- Classifier with default configuration from scipy
        privileged_groups (list)   -- Attribute and corresponding value of privileged group
        unprivileged_groups (list) -- Attribute and corresponding value of unprivileged group
        optim_options (dict)       -- Options if provided by AIF360
    """
    if dataset_used == "adult":
        mutation_strategy  = {"0":[1,0]}
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        dataset_orig = AdultDataset()
    elif dataset_used == "german":
        mutation_strategy = {"1": [0, 1]}
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
        dataset_orig = GermanDataset()
        label_list = dataset_orig.labels
        label_list[label_list == 2] = 0
        dataset_orig.labels = label_list
        dataset_orig.unfavorable_label = 0
    elif dataset_used == "compas":
        mutation_strategy = {"0": [1, 0]}
        if protected == "sex":
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        else:
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        dataset_orig = CompasDataset()
    elif dataset_used == "bank":
        mutation_strategy = {"0": [1, 0]}
        privileged_groups = [{'age': 1}]
        unprivileged_groups = [{'age': 0}]
        dataset_orig = BankDataset()
    elif dataset_used == "mep":
        mutation_strategy = {"0": [1, 0]}
        privileged_groups = [{'RACE': 1}]
        unprivileged_groups = [{'RACE': 0}]
        dataset_orig = MEPSDataset19()

    return dataset_orig, privileged_groups,unprivileged_groups,mutation_strategy