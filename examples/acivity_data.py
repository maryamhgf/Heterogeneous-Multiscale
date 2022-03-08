import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


data = pd.read_csv("/home/mhaghifam/Documents/Research/Neural-ODE/Code/torchdiffeq/examples/WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt", sep="~", header=None)
print(type(data[0]))
print(data[0][0].split(","))
rows = []
for row in data[0]:
    row_seperated = row.split(",")
    row_seperated[-1] = row_seperated[-1].split(";")[0]
    rows.append(row_seperated)

df = pd.DataFrame(rows)
cols = [0, 1, 2, 3, 4, 5]
df = df[df.columns[cols]]
print(df)


def convert_to_int(lst):
    result = []
    j = 0
    for element in lst:
        try:
            result.append(float(element))
        except:
            print(element, j)
            continue
        j = j + 1
    return result

x_acc = convert_to_int(list(df[df.columns[3]]))
y_acc = convert_to_int(list(df[df.columns[4]]))
z_acc = convert_to_int(list(df[df.columns[5]]))
#print(list(df[df.columns[5]]))

dynamics = [x_acc, y_acc, z_acc]
names = ["x", "y", "z"]
i = 0
for dynamic in dynamics:
    print(type(dynamic), len(dynamic))
    f, Pxx = signal.periodogram(dynamic)
    plt.figure()
    plt.plot(f, Pxx)
    plt.title("power spectral density"+names[i])
    plt.savefig("activity2"+names[i]+ "_psd.png")
    i = i + 1
