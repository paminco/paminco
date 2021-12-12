#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
REL_DIR  = "../../../../../"
sys.path.append(REL_DIR)


# # Price of Anarchy - Barcelona

# ## 1. Imports and network readin

# First, we import the *paminco* and some other packages:

# In[2]:


import pandas as pd
import numpy as np

import paminco


# 
# We load the Barcelona network from our converted .xml file. 

# In[3]:


barcelona = paminco.net.Network.from_xml(REL_DIR + "data/traffic/Barcelona.xml")


# Note that we assume that the power factor $p_e$ for the link travel time $l_e(x_e) = \text{fft}_e \cdot \left( 1 + B_e \cdot \left(\frac{x}{\text{cap}_e}\right) ^ {p_e} \right)$ is an integer. This is not the case for the Barcelona network, where we find power factors such  ``4.446, 4.924, ...``.
# 
# However, this allows us to see how we can equip our network with custom cost functions. For this, we first get the link travel time coefficients from the *Barcelona_net.tntp* [[download from here]](https://github.com/bstabler/TransportationNetworks/tree/master/Barcelona) file:

# In[4]:


# https://github.com/bstabler/TransportationNetworks/blob/master/_scripts/parsing%20networks%20in%20Python.ipynb
net = pd.read_csv("data/Barcelona_net.tntp", skiprows=8, sep='\t')
net.columns = [s.strip().lower() for s in net.columns]   
net.drop(['~', ';'], axis=1, inplace=True)

# Get cost coefficients as dict (str -> np.ndarray)
coeffs = net[["free_flow_time", "b", "power", "capacity"]]
coeffs.columns = ["fft", "b", "p", "c"]
coeffs = {k: coeffs[k].values for k in coeffs.columns}


# ## 2. Setting up network cost

# Second, we set up SymbolicCost, defining the edge cost $F_e$ and thus the objective function to calculate both user equilibrium (UE) and system optimum (SO).
# We find minimum cost flows for user equilibrium and system optimum if we transform the edge cost by:
# \begin{align}
# \text{User equilibrium:} \quad F_e &= \int_0^{x_e} l_e(s) ds \\ 
# \text{System optimum:} \quad F_e &= x_e \cdot l_e \\ 
# \end{align}
# 
# Note that [Frank-Wolfe](https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm) requires the first derivative of the objective function.
# Thus, we have to specfiy edge cost $F_e$ and marginal edge cost $f_e$.
# 

# In[5]:


# User equilibrium cost funcs (egde cost and marginal edge cost)
ue_F = "fft * (x + b * x / (p + 1) * (x / c)**p)"
ue_f = "fft * (1 + b * (x / c)**p)"
ue_cost = paminco.net.cost.SymbolicCost(
    coeffs,
    F = ue_F,
    f = ue_f,
    shared=barcelona.shared,
)

# System optimum cost funcs (egde cost and marginal edge cost)
so_F = "x * fft * (1 + b * x**p * c**(-p))"
so_f = "fft + fft * b * (p + 1) * c**(-p) * x**(p)"
so_cost = paminco.net.cost.SymbolicCost(
    coeffs,
    F = so_F,
    f = so_f,
    shared=barcelona.shared,
)


# ## 3. Find UE and SO flow

# Finally, we equip our network with each cost instances and run Frank-Wolfe to find a minimum cost flow:

# In[12]:


barcelona.set_cost(so_cost)
fw_so = paminco.NetworkFW(barcelona)
fw_so.run(max_iter=10, print=False)
fw_so.cost


# For example, we can easily access the user equilibrium flow by:

# In[ ]:


so_cost.cost.save_to_numpy(f)
cost2 = paminco.net.cost.SymbolicCost.from_npz(f)


barcelona.set_cost(so_cost)
fw_so = paminco.NetworkFW(barcelona)
fw_so.run(max_iter=10, print=False)
fw_so.cost


# In[7]:


import tempfile

f = tempfile.mkstemp(suffix=".npz")[1]
barcelona.cost.save_to_numpy(f)
cost2 = paminco.net.cost.SymbolicCost.from_npz(f)

data = np.load(f)
list(data.keys())


# In[8]:


for k,v in data.items():
    print(k)


# In[9]:


cost2 = paminco.net.cost.SymbolicCost.from_npz(f)


# In[10]:


barcelona.cost.save_to_numpy()


# ## 4. Price of Anarchy

# The objective function to calculate the system optimum is the total system travel time (TSTT):
# \begin{equation*}
# \text{TSTT}(\mathbf{x}) = \sum_{e \in E} x_{e} \cdot l_{e}(x_e)
# \end{equation*}

# In[ ]:


def TTST(x):
    return so_cost.F(x).sum()


# We find that the total system travel time for Barcelona is increased by about ``2.3 percent`` if users behave selfishly:

# In[ ]:


poa = TTST(fw_ue.flow) / TTST(fw_so.flow)
poa

