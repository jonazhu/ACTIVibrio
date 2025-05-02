import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from copy import deepcopy

import plot

from binarytree import Node
from scipy.stats import mode

from BinaryTree.binarytree import build_binary_tree, output_binary_tree
from KDTree.kdtree import build_kd_tree
from KDTree.utils import find_subtree, subtree_size, level_order_traversal, level_order_values, get_query_indices

import sys
sys.path.append(".")
from dataloader import read_csv




with open("vc_mutant_data.csv", "r") as fr:
    data = []
    labels = []
    
    # split each line by comma, keep the first value in line as label
    for i, line in enumerate(fr.readlines()):
        if i == 0:
            continue
        vals = line.split(",")
        data.append([float(x) for x in vals[:-1]]) 
        labels.append(vals[-1][:-1]) 
    fr.close()

X = np.array(data)
y = np.array(labels)
unique_labels, numeric_labels = np.unique(y, return_inverse=True)
unique_numeric_labels = np.unique(y)
n_classes = len(unique_labels)

kdtree = build_kd_tree(X, 0, X)

root = Node(kdtree.node)
binary_tree = build_binary_tree(kdtree, root)
output_binary_tree(binary_tree)

def jaccard_index(queries, labels, n_classes):
    clusters = KMeans(n_clusters=n_classes).fit_predict(X[queries])
    jaccard_indices = []
    for i in range(n_classes):
        cluster_indices = np.where(clusters == i)[0]
        cluster_mode = mode(labels[cluster_indices])[0]
        cluster_labels = labels[cluster_indices]
        jaccard_index = len(np.where(cluster_labels == cluster_mode)[0]) / len(cluster_labels)
        jaccard_indices.append(jaccard_index)
    return np.mean(jaccard_indices) * len(queries)

# PLAL: Cluster-based Active Learning
def PLAL(Sx, T, levels, e, sigma, mode="standard", max_depth=np.inf, jaccard_threshold = 50): 
    level = 0
    plal_labels = -np.ones(len(Sx), dtype=np.int32) # initialize array of -1s -> unlabeled values

    active_cells = [[] for _ in range(len(levels))] # maintain a list of active cells for each level
    active_cells[0].append(T) # for the first level, append the root node, which is passed as a parameter into the function
    n_active_indices = len(active_cells[0])
    n_queries = 0
    all_queries = np.array([], dtype=np.int32)
    
    
    high_jaccard_iterations = 0
    
    while n_active_indices > 0:
        
        if mode == "depth":
            if level == max_depth:
                break
        n_active_indices = len(active_cells[level])
        
        qlevel = int((level * 2 * np.log(2) + np.log(1/sigma)) / e) # qlevel parameter, indicates how many values from a particular subtree of the full kdtree should be queried
        n_queries += qlevel
        for i, C in enumerate(active_cells[level]):
            subtree = find_subtree(T, C.node) # locates subtree within full kd tree
            
            subtree_levels = level_order_traversal(subtree, 0, []) # level order traversal of subtree of kdtree
            subtree_values = level_order_values(subtree_levels) # condense level order traversal list into single list of values from subtree            
            query_indices = get_query_indices(subtree_values, qlevel)
            all_queries = pd.unique(np.concatenate([all_queries, query_indices], dtype=np.int32))
            all_query_labels = numeric_labels[all_queries]
            if mode == "jaccard":
                if len(all_queries) > n_classes:
                    jaccard = jaccard_index(all_queries, all_query_labels, n_classes)
                    
                    if jaccard > jaccard_threshold:
                        high_jaccard_iterations += 1
                    if high_jaccard_iterations == 5:
                        return (Sx, plal_labels, n_queries, all_queries)
                
            # tree demo:
            #               x
            #             /   \
            #            y     z
            # level_order_traversal() -> [[x], [y,z]]
            # level_order_values() -> [x,y,z]
            # if qlevel == 2 -> get_query_indices() -> [x,y]

            query_labels = numeric_labels[query_indices]
            
            
            # print_PLAL(active_cells=active_cells, subtree_levels=subtree_levels, qlevel=qlevel, query_indices=query_indices, query_labels=query_labels)


            # condition in PLAL
            # if all labels in query labels are the same -> assign all values in the subtree to consensus label in query_labels
            if len(np.unique(query_labels)) == 1:
                plal_labels[subtree_values] = query_labels[0]
                n_active_indices -= 1
                del subtree 
            elif len(query_indices) < subtree_size(subtree):
                if subtree.right is not None:
                    active_cells[level + 1].append(subtree.right)
                if subtree.left is not None:
                    active_cells[level + 1].append(subtree.left)

        level += 1
    return (Sx, plal_labels, n_queries, all_queries)

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


def active(X, y, inset, outset, seed, max_n_plal, batch_size=10):
    """
    Desc: Active learning main function
    Input: X is input data, y is output labels, inset is current indices used in training set
    Outset is current indices not used in train set, batch_size is number of new instances per round, and random query selection
    Output: Array of accuracy values
    """
    np.random.seed(seed)
    rf = RandomForestClassifier(n_estimators=100, bootstrap=True, min_samples_leaf=9)
    accs = []
    
    i = 0
    # train until 50% of training set is reached (acts as a budget)
    while i < max_n_plal:
        print(f"{len(inset)} / {len(train_indices)}")
        rf.fit(X[inset], y[inset])
        _, y_numeric = np.unique(y, return_inverse=True)
        new_indices = np.random.choice(len(outset), len(outset), replace=False)
        inset = np.append(inset, outset[new_indices[:batch_size]])
        outset = np.delete(outset, new_indices[:batch_size])
        
        acc = cross_validation_accuracy(X[inset], y[inset])
        accs.append(acc)
        i += 1
    
    
    return np.array(accs)

def run_PLAL(X, y, batch_size, seeds):
    """
    Desc: PLAL main function, calculates the best sigma and epsilon parameters using grid search, computes the query labels for PLAL using depth based and index based stopping
    """
    Sx = np.arange(0,len(X))
    kdtree_copy = deepcopy(kdtree) # make copy so we are not overwriting original kdtree
    levels = level_order_traversal(kdtree_copy, 0, [])



    # sigma and epsilon found using grid search for least unlabeled instances above 20% of full dataset
    max_sigma, max_epsilon = 0, 0
    min_unlabeled = np.inf
    for sigma in np.arange(0.1,1,0.1):
        for epsilon in np.arange(0.1,1,0.1):
            _, plal_labels, _, _ = PLAL(Sx, kdtree_copy, levels, sigma, epsilon)
            n_unlabeled = len(np.where(plal_labels == -1)[0])
            # print("Sigma, epsilon, labeled:",sigma, epsilon, n_labeled)
            # print("Sigma, epsilon, unlabeled:",sigma, epsilon, n_unlabeled)
            
            if n_unlabeled < min_unlabeled and n_unlabeled > int(0.2 * len(X)):
                max_sigma = sigma
                max_epsilon = epsilon
                min_unlabeled = n_unlabeled

    _, _, _, depth_queries = PLAL(Sx, kdtree_copy, levels, max_sigma, max_epsilon, "depth", 9)

    _, _, _, jaccard_queries = PLAL(Sx, kdtree_copy, levels, max_sigma, max_epsilon, "jaccard", np.inf, 110)
    
    depth_queries = pd.unique(depth_queries)
    jaccard_queries = pd.unique(jaccard_queries)


    #PLAL accuracies
    depth_accs = []
    jaccard_accs = []

    
    # main loop for calculating accuracy over the depth based and index based stopping queries
    for i in range(5):
        accs = [[],[]]
        print(f"Seed {i+1}")
        np.random.seed(seeds[i])
        for i, queries in enumerate([depth_queries, jaccard_queries]):
        
            rf = RandomForestClassifier(n_estimators=100, bootstrap=True)
            inset = np.array(queries[:batch_size])
            outset = np.array(queries[batch_size:])
            
            while len(inset) < len(queries):
                rf.fit(X[inset], y[inset])
                
                if len(queries) - len(inset) > batch_size:
                    n = batch_size
                else:
                    n = len(queries) - len(inset)
                
                inset = np.append(inset, outset[:n])
                outset = np.delete(outset, range(0,n))
                print(f"{len(inset)} / {len(queries)}")
                acc = cross_validation_accuracy(X[inset], y[inset])
                accs[i].append(acc)
        depth_accs.append(np.array(accs[0]))
        jaccard_accs.append(np.array(accs[1]))
    return depth_accs, jaccard_accs

if __name__ == "__main__":
    X, y = read_csv("../vc_mutant_data.csv")
    unique_labels = np.unique(y)
    train_indices = np.random.choice(len(X), int(0.7*len(X)), replace=False)
    test_indices = np.array([i for i in range(len(X)) if i not in train_indices])


    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    batch_size = 10

    inset = train_indices[np.random.choice(len(train_indices), 10, replace=False)]
    
    
    

    
    seeds = np.random.randint(0, 5000, 5)
    print("Running PLAL - calculating accuracies for index- and depth-based stopping")
    depth_accs, jaccard_accs = run_PLAL(X, y, batch_size, seeds)
    
    print("PLAL Finished.")
    print()
    
    random_accs = []
    
    max_n_plal = max([len(accs) for accs in [depth_accs[0], jaccard_accs[0]]])
    
    while len(np.unique(y[inset])) != len(np.unique(y_train)):
        inset = train_indices[np.random.choice(len(train_indices), 10, replace=False)]
    outset = np.array([x for x in train_indices if x not in inset])
    print("Calculating accuracies via random sampling, stops when same number of indices as PLAL is reached")
    for i in range(5):
        print(f"Seed {i+1}")
        seed = seeds[i]
        random_acc = active(X, y, inset, outset, seed, max_n_plal, batch_size)
        random_accs.append(random_acc)
        
    

    random_accs = np.array(random_accs)
    depth_accs = np.array(depth_accs)
    jaccard_accs = np.array(jaccard_accs) 


    mean_random_acc = np.mean(random_accs, axis=0)
    std_random_acc = np.std(random_accs, axis=0)

    mean_depth_acc = np.mean(depth_accs, axis=0)
    std_depth_acc = np.std(depth_accs, axis=0)

    
    mean_jaccard_acc = np.mean(jaccard_accs, axis=0)
    std_jaccard_acc = np.std(jaccard_accs, axis=0)


    acc_arrays1 = [mean_depth_acc, mean_random_acc]
    std_arrays1 = [std_depth_acc, std_random_acc]

    acc_arrays2 = [mean_jaccard_acc, mean_random_acc]
    std_arrays2 = [std_jaccard_acc, std_random_acc]

    modes1 = ["PLAL - Depth-based stopping", "Random sampling"]
    modes2 = ["PLAL - Index-based stopping", "Random sampling"]
    
    plot.acc(acc_arrays1, std_arrays1, modes1, "plal_depth_stopping")
    plot.acc(acc_arrays2, std_arrays2, modes2, "plal_index_stopping")