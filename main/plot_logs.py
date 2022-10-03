import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import json

fedavg = False

dirs = [f for f in os.listdir("results/") if not f.startswith('.')]

# find the highest number in a list composed by strings that have a number as final char
longest_string = len(max(dirs, key=len))
idx = -2 if longest_string > 5 else -1
highest_number = str(max([int(x[idx:]) for x in dirs if x[idx:].isdigit()]))
#highest_number = "35"

loss_path = "results/run_"+highest_number+"/loss.npy"
acc_path = "results/run_"+highest_number+"/acc.npy"
config_path = "results/run_"+highest_number+"/config.json"

loss_series = np.load(loss_path)
acc_series = np.load(acc_path)

print(loss_series)
print(acc_series)

with open(config_path) as json_file:
    data = json.load(json_file)
    config = {key: val for key, val in data.items()}

if fedavg == False:
    normalized_cm = np.array(config["confusion_matrix"]) / np.array(config["confusion_matrix"]).sum()

    # Create a figure with two sub plots, where the first one plots the loss and the second one the accuracy
    plt.rcParams["figure.figsize"] = (10,5)
    fig, (ax1, ax2_0) = plt.subplots(1, 2)
    fig.suptitle(config["strategy"]+" - m:"+str(config["fraction_mal"])+" - magn:"+str(config["magnitude"]))

    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    sns.heatmap(normalized_cm, vmin=0, vmax=1, annot=True, square=True, cmap='Blues',
        cbar_kws={'format': FuncFormatter(lambda x, _: "%.0f%%" % (x * 100))}, ax=ax1,
        xticklabels='TF', yticklabels='TF')

    ax2_0.set_title("Accuracy")
    lns1 = ax2_0.plot(acc_series, label="accuracy", color="royalblue", )
    ax2_0.set_ylabel("accuracy")
    ax2_0.set_xlabel("epoch")

    ax2_1 = ax2_0.twinx()
    lns2 = ax2_1.plot(config["num_malicious"], label="# malicious", color="lightcoral")

    leg = lns1 + lns2
    labs = [l.get_label() for l in leg]
    ax2_0.legend(leg, labs, loc=0)
    ax2_0.set_facecolor("whitesmoke")
else:
    plt.rcParams["figure.figsize"] = (10,5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle(config["strategy"]+" - m:"+str(config["fraction_mal"])+" - magn:"+str(config["magnitude"]))
    ax.set_title("Accuracy")
    ax.plot(acc_series, label="accuracy", color="royalblue", )
    ax.set_ylabel("accuracy")
    ax.set_xlabel("epoch")
    ax.set_facecolor("whitesmoke")

plt.savefig("results/run_"+highest_number+"/plot.png")
plt.show()