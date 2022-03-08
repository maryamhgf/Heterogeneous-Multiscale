
import pandas as pd
import os
import matplotlib.pyplot as plt

inf_dir = "slow_step100_fast_step5_kernelexp_cutoff3000_nepoch400_fast_nepoch2_fast_step_length1_sensitivityautograd_solverdopri5"
common = inf_dir + "/train_inf.csv"
NODE_file = "./NODE/" + common
NODE_HMM_file = "./NODE_HMM/" + common


data_NODE = pd.read_csv(NODE_file)
data_NODE_HMM = pd.read_csv(NODE_HMM_file)

folder = "./Comparison"
if not os.path.exists(folder):
    os.mkdir(folder)
    print("Directory ", folder,  " Created ")
else:
    print("Directory ", folder,  " already exists")


dirName = "./Comparison/" + inf_dir

if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory ", dirName,  " Created ")
else:
    print("Directory ", dirName,  " already exists")

plt.plot(data_NODE['loss'], label='NODE')
plt.plot(data_NODE_HMM['loss'], label='NODE+HMM')
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")

plt.title("loss")
plt.savefig(dirName + "/loss.png")

plt.figure()
avg_memory = [data_NODE['memory'].mean(), data_NODE_HMM['memory'].mean()]
plt.bar(["NODE", "NODE+HMM"], avg_memory)
plt.ylabel("average allocated memory (MB)")

plt.title("average memory per epoch")
plt.savefig(dirName + "/avg_memory.png")


plt.figure()
avg_memory = [data_NODE['time'].mean(), data_NODE_HMM['time'].mean()]
plt.bar(["NODE", "NODE+HMM"], avg_memory)
plt.ylabel("average allocated time (s)")

plt.title("average time per epoch")
plt.savefig(dirName + "/avg_time.png")
