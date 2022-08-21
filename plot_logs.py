import numpy as np
import os
import matplotlib.pyplot as plt

dirs = os.listdir("results/")

highest_number = str(max([int(x[-1]) for x in dirs if x[-1].isdigit()]))
loss_path = "results/run_"+highest_number+"/loss.npy"
acc_path = "results/run_"+highest_number+"/acc.npy"

loss_series = np.load(loss_path)
acc_series = np.load(acc_path)

print(loss_series)
print(acc_series)

# Create a figure with two sub plots, where the first one plots the loss and the second one the accuracy
plt.rcParams["figure.figsize"] = (10,5)
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
fig.suptitle("Run "+highest_number)
ax1.set_title("Loss")
ax1.plot(loss_series, label="loss", color="royalblue")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")

ax2.set_title("Accuracy")
ax2.plot(acc_series, label="accuracy", color="royalblue")
ax2.set_ylabel("accuracy")
ax2.set_xlabel("epoch")

ax1.set_facecolor("whitesmoke")
ax2.set_facecolor("whitesmoke")

plt.savefig("results/run_"+highest_number+"/plot.png")
plt.show()