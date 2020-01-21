import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

loss = pd.read_csv(sys.argv[1], delimiter=";", header=None)
loss = loss.to_numpy(dtype=np.float)[0]
print(loss)
plt.plot(loss[1:])
plt.show()

