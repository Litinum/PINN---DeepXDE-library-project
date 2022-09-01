import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import math

# delta == w0

# delta = 1
# c1 = 0
# c2 = 2
# df = pd.read_csv("Result_pinn_crit.csv")
3
# delta > w0

# delta = 1.5
# w0 = 1
# alpha = math.sqrt(delta*delta - w0*w0)
# c1 = 0.894427
# c2 = -0.894427
# df = pd.read_csv("Result_pinn_over.csv")

#delta < w0

delta = 0.25
w0 = 1
A = -1.0328
phi = 6.28319*1 + 1.5708
omega = math.sqrt(w0*w0-delta*delta)
df = pd.read_csv("Result_pinn_under.csv")





#xRange = np.linspace(0,10, 222)
xRange = df["t"]
res = []

for x in xRange:
    #y = np.exp(-delta * x)* (c1 + x*c2)        # delta == w0
    #y = np.exp(-delta*x)*(c1*np.exp(alpha*x) + c2*np.exp(-alpha*x))     #  delta > w0
    y = np.exp(-delta*x)*(2*A*math.cos(phi+omega*x))        # delta < w0
    
    res.append(y)
    
#rmse = math.sqrt(mean_squared_error(res, df["y"]))
rmse = np.sqrt((1/len(df["y"])) * np.sum((df["y"] - res)**2))
print(rmse)
    
plt.plot(xRange, res, "r--", df["t"], df["y"], "o")
plt.xlabel("Time [s]")
plt.ylabel("Distance [m]")
#plt.plot(xRange, res)
plt.legend(["Analitical", "PINN"])
plt.show()


