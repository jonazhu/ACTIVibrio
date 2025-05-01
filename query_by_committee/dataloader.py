import numpy as np

"""
File: Dataloader file

Inputs: Filename
Outputs: Reads csv from input filename and stores embedding data in X, label data in y
"""
def read_csv(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        n_lines = len(lines)
        X = np.empty((n_lines, 512))
        y = np.empty(n_lines, dtype=object)
        for i, line in enumerate(lines):
            values = line.split(",")
            X[i] = np.array([float(x) for x in values[:-1]])
            if values[-1][-1] == "\n":
                y[i] = values[-1][:-1]
            else:
                y[i] = values[-1]
        f.close()
    return X, y