import numpy as np

# finds subtree location in kdtree that contains val
def find_subtree(root, val):
    if not root:
        return
    if root.node == val:
        return root
    left_result = find_subtree(root.left, val)
    if left_result is not None:  
        return left_result
    right_result = find_subtree(root.right, val)
    if right_result is not None:  
        return right_result
    
    
# recursively find subtree size
def subtree_size(root):
    if not root:
        return 0
    left = subtree_size(root.left)
    right = subtree_size(root.left)
    return left + right + 1
    
# level order traversal of binary tree
# outputs array of arrays, with each array in full array being list of values in level
def level_order_traversal(root, level, res):
    if not root:
        return

    if len(res) <= level:
        res.append([])
    
    res[level].append(root.node)
    
    level_order_traversal(root.left, level + 1, res)
    level_order_traversal(root.right, level + 1, res)
    return res


# returns array level order traversal values
def level_order_values(levels):
    vals = []
    for level in levels:
        for val in level:
            vals.append(val)
    return np.array(vals)


# subsets level_order_values array up to qlevel
def get_query_indices(values, qlevel):
    indices = []
    for val in values:
        if len(indices) == qlevel:
            break
        indices.append(int(val))
    return np.array(indices, dtype=np.int32)