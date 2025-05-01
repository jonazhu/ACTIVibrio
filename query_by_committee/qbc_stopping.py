import numpy as np
from scipy.special import kl_div

from modAL.models import BayesianOptimizer


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import sys
sys.path.append(".")

import json

import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

import plot
from dataloader import read_csv



def query_by_committee(model, X, y_unique, outset):
    
    """
    Desc: Query by committee active learning function. 
    Input: Takes a trained random forest model, input data, all unique output labels, and all indices in the full training set that havent been encountered yet by the model.
    Output: Returns the indices sorted according to the hard vote entropy of each sample in the outset indices.
    
    Hard vote entropy is computed by argmax -∑ (V(y) / 100) * log(V(y) / 100).
    V(y) is the number of estimators that predicted label y for a specific input sample
    Higher entropy values indicate more committee disagreement, and thus samples with higher entropy should be queried in the next step.
    """
    estimators = model.estimators_
    results = np.zeros((model.n_estimators, len(outset)), dtype = np.int16)
    
    for i, estimator in enumerate(estimators):
        results[i] = estimator.predict(X[outset])
    results = results.T
    
    
    
    # for all predicted labels from the given estimators, count of the number of times each label appears in the set of predicted estimators
    V_y = np.zeros((len(outset), len(y_unique)))
    for i, result in enumerate(results):
        v_y = np.array([])
        for label in y_unique:
            n_label = len(np.where(result == label)[0])
            v_y = np.append(v_y, n_label)
        v_y = np.nan_to_num(v_y)
        V_y[i] = v_y

    # compute the hard vote entropy
    # low entropy means high agreement among estimators
    hard_vote_scores = []
    for val in V_y:
        hard_vote_score = -np.nansum((val / 100) * np.log(val / 100))
        hard_vote_scores.append(hard_vote_score)
    hard_vote_scores = np.array(hard_vote_scores)
    
    # reorder indices based on how high the hard vote entropy of that sample is
    new_indices = np.argsort(hard_vote_scores)[::-1]
    return new_indices, hard_vote_scores


def cross_validation_accuracy(X, y):
    """
    Desc: Computes the cross validation accuracy for the current training set
    Breaks up the current training set into five folds in the inner function "find_folds(n_indices)"
    Then computes the accuracy of the currently trained model on each of the five folds.
    """
    def find_folds(n_indices):
        size = n_indices
        
        indices = np.random.choice(size, size, replace=False)
        folds = np.array_split(indices, 5)
        
        return folds, indices
    cross_val_accuracies = np.array([])
    folds, _ = find_folds(len(X))
    
    for i, fold in enumerate(folds):
        model = RandomForestClassifier(n_estimators=100)
        if len(folds[:i]) == 0:
            train_folds = np.concatenate(folds[i+1:])
        elif len(folds[i+1:]) == 0:
            train_folds = np.concatenate(folds[:i])
        else:
            train_folds = [folds[index] for index in range(len(folds)) if index != i]
            train_folds = np.concatenate(train_folds)
        model.fit(X[train_folds], y[train_folds])
        fold_pred = model.predict(X[fold])
        cross_val_accuracies = np.append(cross_val_accuracies, accuracy_score(fold_pred, y[fold]))
    return np.mean(cross_val_accuracies)
    
    
def pad_arrays_with_nan(arrays):
    """
    Desc: Array padding function
    Different seeds result in different length arrays due to randomness and stopping criteria.
    This function unifies the size of each array to the maximum array size by padding with nan values.
    """
    # Find the length of the longest array
    max_length = max(len(arr) for arr in arrays)
    
    # Pad each array to the max length with np.nan
    padded_arrays = []
    for arr in arrays:
        # Create padded array with np.nan values
        padded = np.full(max_length, np.nan)
        # Copy the original array values to the beginning of the padded array
        padded[:len(arr)] = arr
        padded_arrays.append(padded)
    
    return padded_arrays



def committee_agreement_stopping(model, qbc_args, stagnant_states):    
    """
    Desc: Stopping based on committee disagreement
    Inputs: random forest model trained on current dataset up until this point, arguments for when to stop querying new points,
    the current number of consecutive iterations where disagreement was below threshold
    Outputs: Number of consecutive iterations that were stagnating
    """
    tree_probs = np.zeros((len(model.estimators_), X[outset].shape[0], 9))
    for i, estimator in enumerate(model.estimators_):
        tree_probs[i] = estimator.predict_proba(X[outset])
    rf_probs = model.predict_proba(X[outset])
    
    eps = qbc_args["eps"]
    kl_threshold = qbc_args["kl_threshold"]
    
    divs = []
    for probs in tree_probs:
        divs.append(kl_div(rf_probs+eps, probs+eps))
    avg_kl = np.mean(divs)
    
    if avg_kl < kl_threshold:
        stagnant_states += 1
    else:
        stagnant_states = 0
    
    return stagnant_states
def active(X, y, inset, outset, qbc_args, seed, batch_size=10, mode="random"):
    """
    Desc: Active learning main function
    Input: X is input data, y is output labels, inset is current indices used in training set
    Outset is current indices not used in train set, batch_size is number of new instances per round, and a query selection mode
    Output: Array of accuracy values
    """
    np.random.seed(seed)
    rf = RandomForestClassifier(n_estimators=100, bootstrap=True, min_samples_leaf=9)
    accs = []
    

    stagnant_states = 0
    
    # train until 50% of training set is reached (acts as a budget)
    while len(inset) < int(0.5*len(train_indices)):
        print(f"{len(inset)} / {len(train_indices)}")
        rf.fit(X[inset], y[inset])
        _, y_numeric = np.unique(y, return_inverse=True)
        y_numeric_unique = np.unique(y_numeric)
        if mode == "qbc":
            new_indices, _ = query_by_committee(rf, X, y_numeric_unique, outset)
            # bootstrap_correlation(rf, X, outset)
        elif mode == "random":
            new_indices = np.random.choice(len(outset), len(outset), replace=False)
        
        inset = np.append(inset, outset[new_indices[:batch_size]])
        outset = np.delete(outset, new_indices[:batch_size])
        
        
        stagnant_states = committee_agreement_stopping(rf, qbc_args, stagnant_states)

        acc = cross_validation_accuracy(X[inset], y[inset])
        # acc = accuracy_score(y_pred, y_test)
        accs.append(acc)
        
        # if the predicted tree probs does not differ much from the full random forest probs for 5 iterations, then break
        if stagnant_states == 5:
            break
        
    return np.array(accs)
    
        

if __name__ == "__main__":
    batch_size = 10
    
    X, y = read_csv("../vc_mutant_data.csv")
    unique_labels = np.unique(y)
    train_indices = np.random.choice(len(X), int(0.7*len(X)), replace=False)
    test_indices = np.array([i for i in range(len(X)) if i not in train_indices])


    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]


    inset = train_indices[np.random.choice(len(train_indices), 10, replace=False)]
    while len(np.unique(y[inset])) != len(np.unique(y_train)):
        inset = train_indices[np.random.choice(len(train_indices), 10, replace=False)]
    outset = np.array([x for x in train_indices if x not in inset])


    # base random forest model, offline model to serve as comparison
    base_rf = RandomForestClassifier(n_estimators=100)
    base_rf.fit(X_train, y_train)
    y_pred = base_rf.predict(X_test)
    target_acc = accuracy_score(y_pred, y_test)
    cm = confusion_matrix(y_pred, y_test)
    cm_ratios = np.round(cm / np.sum(cm, axis=0), 3)



    with open("qbc_args.json", "r") as f:
        qbc_args = json.load(f)
        f.close()
    
    
    qbc_accs = []
    random_accs = []
    
    # main training loop - run 5 seeds, calculate mean and std accuracy across all 5 seeds for query by committee and random sampling
    for i in range(5):
        print(f"Seed {i+1}")
        seed = np.random.randint(5000)
        qbc_acc = active(X, y, inset, outset, qbc_args, seed, batch_size, "qbc")
        random_acc = active(X, y, inset, outset, qbc_args, seed, batch_size, "random")
        
        qbc_accs.append(qbc_acc)
        random_accs.append(random_acc)
    
    qbc_accs = pad_arrays_with_nan(qbc_accs)
    qbc_accs = np.array(qbc_accs)
    
    
    random_accs = pad_arrays_with_nan(random_accs)
    random_accs = np.array(random_accs)

    mean_qbc_accuracy = np.mean(qbc_accs, axis=0)
    std_qbc_accuracy = np.std(qbc_accs, axis=0)
    
    mean_random_accuracy = np.mean(random_accs, axis=0)
    std_random_accuracy = np.std(random_accs, axis=0)


    mean_qbc_accuracy = mean_qbc_accuracy[~np.isnan(mean_qbc_accuracy)]
    std_qbc_accuracy = std_qbc_accuracy[~np.isnan(std_qbc_accuracy)]
    mean_random_accuracy = mean_random_accuracy[~np.isnan(mean_random_accuracy)]
    std_random_accuracy = std_random_accuracy[~np.isnan(std_random_accuracy)]
    
    
    
    acc_arrays = [mean_qbc_accuracy, mean_random_accuracy]
    std_arrays = [std_qbc_accuracy, std_random_accuracy]
    modes = ["Query by committee", "Random sampling"]
    plot.acc(acc_arrays, std_arrays, modes)