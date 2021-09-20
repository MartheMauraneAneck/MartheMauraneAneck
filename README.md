# Marthe Maurane Aneck
# CST-305
# Project 1
# 09/19/2021

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dx/dt
def model(x,t,k):
    dxdt = k * x
    return dxdt

# Our initial condition where we have 6 people using the same wifi network
x0 = 6

# time points
t = np.linspace(0,20)

# solvig ODEs based on our assumption
# k represent the amount of search on queue using the network, the higher k the higher the queue
k = 0.2
x1 = odeint(model,x0,t,args=(k,))
k = 0.5
x2 = odeint(model,x0,t,args=(k,))
k = 0.7
x3 = odeint(model,x0,t,args=(k,))

# ploting our  results
plt.plot(t,x1,'r-',linewidth=2,label='k=0.2')
plt.plot(t,x2,'b--',linewidth=2,label='k=0.5')
plt.plot(t,x3,'g:',linewidth=2,label='k=0.7')
plt.xlabel('time')
plt.ylabel('x(t)')
plt.legend()
plt.show()
