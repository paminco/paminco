#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
REL_DIR  = "../../../.."
sys.path.append(REL_DIR)


# # Setting up a Demand Function

# ## 1. Why do we need it?

# paminco allows to compute parametric minimum cost flows (MCF) w.r.t. to demand multiplier $\lambda$. I.e., the goal is to find a MCF for all values of $\lambda$ in the interval $\left[0, \lambda^{\max }\right]$:
# 
# \begin{align*}
# \min \text { } & C(\mathbf{x}) \\
# \text {s.t.} \quad \mathbf{\Gamma} \mathbf{x} &= \mathbf{h}(\lambda), \\
# \mathbf{x} &\geq \mathbf{0},
# \end{align*}
# 
# where $\mathbf{h}(\lambda)$ is a ``demand function`` that maps $\lambda$ to a ``demand vector`` $\mathbf{b}$.

# ## 2. Demand vectors and matrices 

# ### 2.1. Demand vector

# For single commodity problems a demand vector is a (n, ) array that encodes the inflow and outflow rate per node, negative values indicating *sources* and positive values *sinks*.
# 
# Consider the following example. 
# The network consists of four vertices and five edges. 
# We want to send 10 units from vertex $A$ to vertex $D$, indicated by the wiggly green arrow
# <center> 
# <img src="img/graph_demand.jpg" width="400">
# </center>

# #### Network setup

# We setup this network (without demand) as follows:

# In[2]:


import paminco
import numpy as np

edge_data = np.array([
    ["A", "B"],
    ["A", "C"],
    ["B", "C"],
    ["B", "D"],
    ["C", "D"],
])
graph = paminco.Network(edge_data)
graph.nodes.lbl2id


# #### Building demand vector

# We now build a demand vector:

# In[3]:


demand_vector = paminco.net.demand.demand_vector(
    graph.shared,  # each network element must know about shared data ...
    ("A", "D", 10)  # send 10 units from A to D
)
demand_vector.map_node_label_to_id()


# #### Retrieving sparse representation

# This demand vector consists of one CommoditySingleSourceSink:

# In[4]:


demand_vector.commodities


# Algorithms in paminco expect the demand vector to be sparse matrix, which can be accessed by:

# In[5]:


demand_vector()


# In[6]:


demand_vector().toarray()


# #### Setting demand function

# We equip the network with a linear demand function that scales this demand vector by the demand multiplier $\lambda$:

# In[23]:


graph.set_demand(demand_vector)
graph.demand


# For a linear demand function with $\lambda = 1$ (no scaling), we expect to retrieve the specified demand vector:

# In[24]:


graph.demand(1).toarray()


# For other values of $\lambda$, the input demand vector is scaled:

# In[25]:


graph.demand(0.32).toarray()


# In[10]:


graph.demand(1.63).toarray()


# #### All in one go

# By default, the network constructor builds a linear demand function from the ``demand_data``: 

# In[11]:


graph.set_demand(("A", "D", 10))
graph.demand(1.63).toarray()


# ### 2.2. Demand matrix

# For certain problems, the demand may consist of multiple commodities. 
# E.g., in a traffic network some drivers want to go from one location to another, and others vice versa.
# Consider an adaption of the graph above:
# <center> 
# <img src="img/graph_demand_multi.jpg" width="400">
# </center>
# 
# Here, we set five commodities
# \begin{align*}
# A & \overset{2}{\rightarrow} B \quad (\text{blue}), \\
# A & \overset{5}{\rightarrow} D \quad (\text{orange}), \\
# B & \overset{7}{\rightarrow} A \quad (\text{light green}), \\
# C & \overset{5}{\rightarrow} B \quad (\text{dark green}), \\
# D & \overset{5}{\rightarrow} C \quad (\text{red}).
# \end{align*}
# 
# <!-- A \rightarrow B $ (blue), $A \rightarrow D $ (orange), $B \rightarrow A $ (light green), $C \rightarrow B $ (dark green), $D \rightarrow C $ (red). -->
# <!-- $A \overset{2}{\rightarrow} B $ (blue),
# $A \overset{5}{\rightarrow} D $ (orange),
# $B \overset{7}{\rightarrow} A $ (light green),
# $C \overset{5}{\rightarrow} B $ (dark green),
# $D \overset{5}{\rightarrow} C $ (red). -->

# In[12]:


commodities = [
    ("A", "B", 2),
    ("A", "D", 5),
    ("B", "A", 7),
    ("C", "B", 5),
    ("D", "C", 5),
]
demand_matrix = paminco.net.demand_vector(graph.shared, commodities)
demand_matrix.map_node_label_to_id()


# We encoded the commodites in a demand matrix, where each column is a demand vector:

# In[13]:


import pandas as pd
d = pd.DataFrame(demand_matrix().toarray())
d.set_index(graph.nodes.labels)


# ### 2.3. Commodity with multiple sinks and sources

# So far, we considered only commodities that have a single source and single sink. 
# However, for certain applications -- such as a gas network -- a commodity may consits of ``multiple sources and/or multiple sinks``.
# This CommodityMultiSourceSink is encoded as a vector or dictionary that specfiies in and outflow rate of each node:

# In[14]:


com = {"A": -5, "B": -2, "C": 1, "D": 6}
demand_vector = paminco.net.demand_vector(graph.shared, com)
demand_vector.commodities


# In a sense, hhis commodity specifies that we input 5 units at $A$ and 2 at $B$, which are consumed by $D (6)$ and $C (1)$, i.e., $(A, B) \overset{7}{\rightarrow} (C, D)$.'

# ## 3. Function types

# paminco has allows for two demand function types (other can be implemented by inheritance of paminco.demand.DemandFunction): linear demand functions and affine demand functions. 
# For notational simplicity, we only consider demands that consists of a single commodity for the rest of this section.

# ### 3.1. Linear demand function

# A linear demand function is of the form:
# \begin{equation}
# \mathbf{h}_{\text{lin}}(\lambda) = \lambda \mathbf{b}, 
# \end{equation}
# 
# where $\mathbf{b}$ is a demand vector (see 1.1). In a sense, the demand is simply scaled by the parameter $\lambda$.
# 
# 

# In[15]:


import paminco
import numpy as np

edge_data = np.array([
    ["A", "B"],
    ["A", "C"],
    ["B", "C"],
    ["B", "D"],
    ["C", "D"],
])
demand_data = ("A", "D", 1)

graph = paminco.Network(edge_data, demand_data=demand_data)


# The demand function of the network for $\lambda = 2.5$ -- i.e., sending 2.5 times the original demand (1) from $A$ to $D$ -- is called by:

# In[16]:


graph.h(2.5)


# which is a $n \times k$ sparse matrix. This matrix encodes the in and outflow rates for all nodes (row) for all commodities (column). It is easily converted to a numpy.ndarray:

# In[17]:


graph.h(2.5).toarray()


# For example, the -2.5 indicates that the node at index 0 (node $A$) has a total inflow of -2.5 units, negative rates thus marking *source*. All the inflow in node $A$ is then "consumed" by the node at index 3 (node $D$, a *sink*).

# ### 3.2. Affine demand function

# In contrast, an *affine demand function* consists of two parts: a base demand $\mathbf{b}_0$ that is invariant of $\lambda$ and a part that scales linearly with the demand multiplier:
# \begin{equation}
# \mathbf{h}_{\text{affine}}(\lambda) = \mathbf{b}_0 + \lambda \mathbf{b}.
# \end{equation}
# 
# This is of interest for problems where there exsists a trivial solution to problems with linearly scaled demands.
# Using the data from the above example, we can setup an affine demand function as follows:

# 

# In[18]:


b0 = ("B", "D", 1)
b = ("A", "D", 1)
demand_data = (b0, b)

graph2 = paminco.Network(
    edge_data, 
    demand_data=demand_data, 
    kw_demand={"mode": "affine"}
)


# This graph now has a demand part that is independent of the demand multiplier:

# In[19]:


graph2.h(0).toarray()


# and one that scales linearly with it:

# In[20]:


graph2.h(7).toarray()


# We can think of calling this affine demand function with $\lambda = 7$ as sending 7 units from $A$ to $D$ and 1 unit from $B$ to $D$, resuilting in total 8 units in the sink $D$.
