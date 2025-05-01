import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def cm(cm_ratios, labels):
    
    fig, ax = plt.subplots()

    italicized_labels = []
    for label in labels:
        underscore_loc = label.find("_")
        if label == "WT":
            italicized_labels.append(label)
            continue
        
        elif underscore_loc != -1:
            italicized_labels.append("$\it{{{0}}}\_{1}$".format(label[:underscore_loc], label[underscore_loc + 1:]))
        else:
            if len(label) > 5:
                italicized_labels.append(label)
                continue
            italicized_labels.append("$\it{{\Delta {0}}}$".format(label))



    sns.heatmap(cm_ratios, cmap="gray_r", fmt=".1%", annot=cm_ratios, cbar=False)
    plt.rc('axes', labelsize=10)    # fontsize of the x and y labels

    ax.set_xlabel("Actual Labels")
    ax.set_ylabel("Predicted Labels")
    ax.set_xticklabels(italicized_labels, rotation=45)
    ax.set_yticklabels(italicized_labels, rotation=360)
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    # ax.patch.set_linewidth(1)
    plt.show()
    
    
def acc(acc_arrays, std_arrays, modes):
    color_palette = sns.color_palette("tab10")    
    plt.figure(figsize=(12,8))
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
    

    
    plt.savefig("qbc_stopping_plot.png")
    plt.show()
    