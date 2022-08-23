import numpy as np
import os
import matplotlib.pyplot as plt
import json

dirs = os.listdir("results/")

highest_number = str(max([int(x[-1]) for x in dirs if x[-1].isdigit()]))
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
config["num_clients"].insert(0, 0)  # first round is empty
config["num_malicious"].insert(0, 0.0)  # first round is empty

# Create a figure with two sub plots, where the first one plots the loss and the second one the accuracy
plt.rcParams["figure.figsize"] = (10,5)
fig, (ax1, ax2_0) = plt.subplots(1, 2, sharex=True)
fig.suptitle(config["strategy"]+" - m:"+str(config["fraction_mal"])+" - magn:"+str(config["magnitude"]))

ax1.set_title("Loss")
ax1.plot(loss_series, label="loss", color="royalblue")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")

ax2_0.set_title("Accuracy")
lns1 = ax2_0.plot(acc_series, label="accuracy", color="royalblue")
ax2_0.set_ylabel("accuracy")
ax2_0.set_xlabel("epoch")

ax2_1 = ax2_0.twinx()
lns2 = ax2_1.plot(config["num_clients"], label="# clients", color="lightsteelblue")
ax2_1.set_ylabel("# clients")

lns3 = ax2_1.plot(config["num_malicious"], label="# malicious", color="lightcoral")

leg = lns1 + lns2 + lns3
labs = [l.get_label() for l in leg]
ax2_0.legend(leg, labs, loc=0)

ax1.set_facecolor("whitesmoke")
ax2_0.set_facecolor("whitesmoke")

plt.savefig("results/run_"+highest_number+"/plot.png")
plt.show()