import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('report/SAC_7.csv')

fig, ax = plt.subplots()
ax.plot(data.values[:,1], data.values[:,2])
plt.show()