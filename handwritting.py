#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


data_file = open("mnist_train_100.csv", "r")
data_list = data_file.readlines()
data_file.close()
print(data_list)

plt.plot(data_list)
plt.show()
plt.close()
"""
for i in range(len(data_list)):
    all_vlaues = data_list[i].split(",")
    x = np.asfarray(all_vlaues[i])*10.0

"""    
