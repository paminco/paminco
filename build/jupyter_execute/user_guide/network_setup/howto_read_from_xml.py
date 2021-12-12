#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
REL_DIR  = "../../../.."
sys.path.append(REL_DIR)


# # How to read data from XML

# For paminco, we chose XML as our standard data format for networks. An XML file specifies edges (obligatory), nodes (optional), edge cost (optional) and commodities (optional). 

# ## 1. Read edges and Nodes

# ### 1.1 From -> to

# We start by specyfying and reading in only the egde data. 

# In[2]:


import paminco

xml1 = """
<network>
  <edges>
    <edge from="A" to="B"></edge>
    <edge from="A" to="C"></edge>
    <edge from="B" to="C"></edge>
    <edge from="B" to="D"></edge>
    <edge from="C" to="D"></edge>
  </edges>
</network>
"""
edges = paminco.net.Edges.from_xml(xml1)
edges.to_df()


# ### 1.2 Edge bounds

# By default, paminco constructs directed edges with bounds ``(0, infinity)``. 
# However, we can specify them in the XML file:

# In[3]:


xml2 = """
<network>
  <edges>
    <edge from="A" to="B" lb="0" ub="200"></edge>
    <edge from="A" to="C" lb="-200" ub="200"></edge>
    <edge from="B" to="C" lb="10" ub="11"></edge>
    <edge from="B" to="D" lb="1" ub="1.1"></edge>
    <edge from="C" to="D" lb="-5" ub="3"></edge>
  </edges>
</network>
"""
edges = paminco.net.Edges.from_xml(xml2)
edges.to_df()


# ### 1.3 Node attributes

# We can add a ``<nodes>`` tag to the XML and define node locations:

# In[4]:


xml3 = """
<network>
  <nodes>
    <node node="A" x="0" y="0"/>
    <node node="B" x="0" y="3"/>
    <node node="C" x="3" y="0"/>
    <node node="D" x="3" y="3"/>
  </nodes>
  <edges>
    <edge from="A" to="B"></edge>
    <edge from="A" to="C"></edge>
    <edge from="B" to="C"></edge>
    <edge from="B" to="D"></edge>
    <edge from="C" to="D"></edge>
  </edges>
</network>
"""


# In[5]:


nodes = paminco.net.Nodes.from_xml(xml3)
nodes.to_df()


# ## 2. Read commodities

# In[6]:


xml4 = """
<network>
  <commodities>
    <commodity>
      <b node="1" value="-160.0"/>
      <b node="2" value="-60.0"/>
      <b node="7" value="100.0"/>
      <b node="14" value="120.0"/>
    </commodity>
  </commodities>
  <edges>
    <edge from="A" to="B"></edge>
    <edge from="A" to="C"></edge>
    <edge from="B" to="C"></edge>
    <edge from="B" to="D"></edge>
    <edge from="C" to="D"></edge>
  </edges>
</network>
"""

dv = paminco.net.demand_vector(xml4, is_label=False)
dv.commodities[0]


# In[7]:


a = {
    1 : -160.0,
2 : -60.0,
7 : 100.0,
14 : 120.0,
}


# In[8]:


labels = ["A", "B", "C"]
indices = [0, 1, 2]
rates = [8, 17, -25]

out = ""
for i in range(3):
    out += f"\n'{labels[i]}' ({indices[i]}): {rates[i]}"
    
print(out)


# In[9]:


dv = paminco.net.demand_vector(xml4, is_label=False)
self = dv.commodities[0]
items = ("'%s' : %r" % (k, v) for k, v in self.to_dict().items())
'\n'.join(items)
self.to_dict().items()


# In[10]:


from paminco.net.shared import Shared, ID_UNMAPPED, LBL_UNMAPPED
LBL_UNMAPPED in self._node_labels


# ## 3. Read edge cost

# ### 3.1 Polynomial cost

# ### 3.2 Piecewise quadratic cost

# ## 4. Read the whole network
