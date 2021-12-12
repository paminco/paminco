#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
REL_DIR  = "../../../../../"
sys.path.append(REL_DIR)


# # Sioux Falls - User Equilibrium

# ## 1. Imports and data readin

# First we load the paminco package and read the SiouxFalls data. 
# The data was taken from [here](https://github.com/bstabler/TransportationNetworks) and converted to our format of choice: ``XML``.
# The data for SiouxFalls comes with paminco and can be easily loaded: 

# In[2]:


import paminco

sioux = paminco.load_sioux()


# By default the edge cost equal the link travel time: $F_e = l_e$. 
# The link travel time is defined as
# \begin{equation*}
# l_e(x_e) = \text{fft}_e \cdot \left( 1 + B_e \cdot \left(\frac{x}{\text{cap}_e}\right) ^ {p_e} \right)
# \end{equation*}

# ## 2. Calculating the user equilibrium flow with Frank-Wolfe

# Now we can use the Frank-Wolfe optimizer to find an approximate minimum cost flow that conincides <
# with the user equilibrium. 
# To achive this, we have to integrate the edge costs by:
# \begin{equation*}
# F_e = \int_0^{x_e} l_e(s) ds
# \end{equation*}

# In[3]:


sioux.integrate_cost()
fw = paminco.NetworkFW(sioux)
fw.run(max_iter=200)


# The UE flow can be accessed by:

# In[4]:


fw.flow


# ## 3. Plotting with NetworkX

# Now we use the plotting capabilities of network to plot the equilibrium flow.
# 
# First, we build a nx.DiGraph from out network:

# In[5]:


import matplotlib.pyplot as plt
import networkx as nx

# Get edgelist (pd.DataFrame) and pos dict
edgelist = sioux.get_flow_df(fw.flow)
pos = sioux.get_node_pos()

# Select only edge with flow
edgelist = edgelist[edgelist.flow > 0].reset_index()
G = nx.from_pandas_edgelist(
    edgelist,
    edge_attr=["flow"],
    create_using=nx.DiGraph(),
)


# We setup a canves and plot the nodes (with labels) and the user equilibrium flow:

# In[6]:


plt.figure(figsize=(10, 14))
nx.draw_networkx_nodes(G, pos, node_color="lightgrey", edgecolors="black", alpha=0.5)
nx.draw_networkx_edges(
    G, 
    pos, 
    width=2*edgelist.flow/edgelist.flow.max() + 0.2,
    connectionstyle="arc3, rad=0.1"
)
_  = nx.draw_networkx_labels(G, pos)

