#!/usr/bin/env python
# coding: utf-8

# # Graph Setup and Representations

# Denote $G = (V, E)$ a ``graph``, where $V$ are the vertices or nodes and $E$ are the links or edges. 
# This notebook shows how to setup directed and undirected graphs, how to access different graph respresentations 
# and how to specify labels to indices mappings. 

# ## 1 Directed graphs

# A ``directed graph`` is characterized by the direction of the edge flow, i.e., units can only travel along the edge in one direction.
# Consider the following directed graph:
# <center> 
# <img src="img/dir_graph.jpg" width="500">
# </center>
# 
# that connectects the four vertices $A, B, C, D$ with the five directed links $e_1, e_2, e_3, e_4$ and  $e_5$.

# ### 1.1 Setup

# We can setup the above graph with paminco:

# In[2]:


import sys
REL_DIR  = "../../../.."
sys.path.append(REL_DIR)

import numpy as np
import paminco

edge_data = np.array([
    ["A", "B"],
    ["A", "C"],
    ["B", "C"],
    ["B", "D"],
    ["C", "D"],
])

graph = paminco.Network(edge_data, kw_edge={"directed": False})
graph.edges.to_df()


# ### 1.2 Representations

# A graph can be represented by its $n \times n$ adjacency matrix, which store as a sparse matrix:

# In[ ]:


graph.adjacency_matrix().toarray()


# or its $n \times m$ incidence matrix

# In[ ]:


graph.incidence_matrix().toarray()


# #### 1.2.1 Adjacency matrix

# #### 1.2.2 Incidence matrix

# #### 1.2.3 Laplacian matrix

# ### 1.3 Edge bounds

# Depending on the specific problem, each edge may only allow a certain maximum of flow on the edge, i.e., it has an ``upper bound``. By default these bounds are $(0, \infty)$ for directed edges. However, it is possible to pass individual flow bounds for each edge:

# In[ ]:


edge_labels = np.array([
    ["A", "B"],
    ["A", "C"],
    ["B", "C"],
    ["B", "D"],
    ["C", "D"],
])
edge_bounds = np.array([
    [1, 100],
    [4, 5],
    [300, 400],
    [0, 80000],
    [12, 12.1],
])
edge_data = (edge_labels, edge_bounds)

graph = paminco.Network(edge_data)
graph.edges.to_df()


# ## 2 Undirected graphs

# ### 2.1 Setup

# ### 2.2 Representations

# #### 2.2.1 Adjacency matrix

# #### 2.2.2 Incidence matrix

# #### 2.2.3 Laplacian matrix

# ### 2.3 Edge bounds

# In[11]:


graph.csgraph(respect_bounds=False).toarray()


# In[10]:


graph.csgraph(respect_bounds=False).toarray()


# ## 3. Labels to indices mapping

# In a network, the ``nodes`` objects handles the mapping of labels to indices, which influences the graph representation.
# The mapping is accessed by:

# In[6]:


graph.nodes.lbl2id


# It is possible to set a mapping in two ways:
# 
# 1) pass in indices in node data
# 2) specfiy mapping for edge constructor

# ### 3.1 Pass indices in node data

# In[8]:


labels = ["A", "B", "C", "D"]
indices = [3, 1, 0, 2]
zone = False
xy = [(0, 0), (4, 3), (8, 8), (1, 9)]
node_data = (labels, indices, xy, zone)

g = paminco.Network(edge_data, node_data)
g.nodes.lbl2id


# E.g., compare the indicence matrices, where the specified mapping changes to row order:

# In[10]:


g.incidence_matrix().toarray()


# In[11]:


graph.incidence_matrix().toarray()


# For more information see documentation of Nodes:

# In[7]:


print(paminco.net.Nodes.__doc__[:1010])  # smaller version of paminco.net.Nodes?


# ### 3.2 Specifing label mapping in edges

# We can achieve the same by specfying a ``map_labels_to_indices`` mapping in edges:

# In[21]:


edge_data = np.array([
    ["A", "B"],
    ["A", "C"],
    ["B", "C"],
    ["B", "D"],
    ["C", "D"],
])
mapping = {'C': 0, 'B': 1, 'D': 2, 'A': 3}
g2 = paminco.Network(edge_data, kw_edge={"map_labels_to_indices": mapping}, dtype_int=np.int64)
g2.incidence_matrix().toarray() - g.incidence_matrix().toarray()

