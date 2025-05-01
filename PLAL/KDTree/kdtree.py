import numpy as np

def median(data, split_dim):
    '''
    Find the median of the data at the given split dimension.
    Inputs: NxD data where N is the number of samples and D is the number of dimensions. split_dim is the dimension to split on for this iteration of building out the kd tree
    Outputs: The index of the median of the values in the data at the column of the correct split dimension.
    '''
    return np.argsort(data, axis=0)[int(len(data) / 2), split_dim]

# check index of row in matrix in numpy
def check_index(array, matrix):
    '''
    Checks the index of the current split row in the primary main dataset.
    Dictates location to split into left and right subsets of data matrix.
    
    Input: full row of data in the primary data matrix.
    Output: Index of row in primary data matrix
    '''
    for i, row in enumerate(matrix):
        if np.array_equal(array, row):
            return i
    return -1
class KDTree:
    '''
    KDTree class, basic binary tree structure that iterates through dimensions, splitting into two children on each dimension.
    '''
    def __init__(self, node, indices, left=None, right=None):
        self.node = node
        self.indices = indices
        self.left = left
        self.right = right
def build_kd_tree(data, split_dim, X):
    '''
    Basic kdtree construction function. 
    Keeps track of the current split dimension. If the split dimension reaches the max # of columns, then split on the first dimension again.
    Increment every time a split is performed.
    Split is dictated by median of the current split dimension column.
    Dataset is broken into two subsets at each split: left and right.
    Left and right subsets are split recursively until there are no more splits left to perform (length of data -> 0) 
    '''
    if split_dim == data.shape[1]:
        split_dim = 0 
    if len(data) == 0: # base case -> data is empty -> not possible to split data by median further
        return
    
    node = median(data, split_dim)
    split_dim += 1
    
    left_data = data[:node]
    right_data = data[node + 1:]
    
    # recursive split dataset by next dimension until cant split anymore
    return KDTree(check_index(data[node], X), 
                  split_dim,
                  left=build_kd_tree(left_data, split_dim, X),
                  right=build_kd_tree(right_data, split_dim, X))