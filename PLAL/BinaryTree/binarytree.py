from binarytree import Node


def build_binary_tree(tree, root=None):
    if tree is not None:
        if tree.left is not None:
            root.left = Node(tree.left.node)
        if tree.right is not None:
            root.right = Node(tree.right.node)
        build_binary_tree(tree.left, root.left)
        build_binary_tree(tree.right, root.right)
    return root



def output_binary_tree(binary_tree):
    with open("space_partitioning_tree.txt", "w") as f:
        print(binary_tree, file=f)
        f.close()