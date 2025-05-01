import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def acc(acc_arrays, std_arrays, modes, name):
    color_palette = sns.color_palette("tab10")    
    plt.figure(figsize=(8,5))
    # fig, ax = plt.subplots(len(acc_arrays), 1, figsize=(5.5*len(acc_arrays), 15))
    
    for i, (accs, stds) in enumerate(zip(acc_arrays, std_arrays)):
        x = np.arange(0, len(accs))
        y = accs
        plt.plot(x, y, color=color_palette[i], label=modes[i])
        plt.fill_between(x, accs - stds, accs + stds, alpha = 0.2, color=color_palette[i])
        plt.fill_between(x, accs - stds, accs + stds, alpha = 0.2, color=color_palette[i])
    plt.legend()
    
    plt.xlabel("Iteration (batches of 10)")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(0, max([len(accs) for accs in acc_arrays]), 1))
    

    
    plt.savefig(f"{name}.png")
    plt.show()
    