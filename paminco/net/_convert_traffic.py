import re
import numbers

import numpy as np
import pandas as pd


LINK_TRAVEL_TIME = "free_flow_time * (1 + b * (x / capacity)**power)"


def dummy_file_handler(file):
    if not hasattr(file, 'read'):
        file = open(file, 'r')
    return file


def clever_split(line: str) -> list:
    """Get separated (tabs or whitespaces) elements from line.

    Parameters
    ----------
    line : str

    Returns
    -------
    elems : list of str
        Separated elements.
    """
    # split elements by tabs
    cols = line.split('\t')

    # if no tabs, split by whitespace
    if len(cols) <= 1:
        cols = line.split(' ')

    # strip elements of trailing whitespaces
    elems = [c.strip() for c in cols if len(c.strip()) > 0]

    return elems


def load_trips(trips) -> list:
    """Read network commodities from path.
    
    Parameters
    ----------
    tntp_trips : str
        File path.
    
    Returns
    -------
    commodities : list of list
        List of commodities (source, target, demand).
    """
    # Open file
    f = dummy_file_handler(trips)
    
    # Init return data
    commodities = []
    
    try:
        for line in f:
            line = line.strip()
            
            # skip emtpy lines
            if len(line) == 0:
                continue
            
            # read metadata from line if availabe
            m = re.findall(r'\<([\w\s]+)\>\s*(\w+)\s*', line)
            if len(m) > 0:
                continue
            
            cols = clever_split(line)
            
            # update current origin
            if len(cols) > 0 and cols[0].strip().lower() == 'origin':
                origin = cols[1].strip()
                continue
            
            # read commodity data
            for c in line.split(';'):
                data = c.strip().split(':')
                if len(data) > 1:
                    target, load = data[0].strip(), data[1].strip()
                    load = float(load)
                    commodities.append((origin, target, load))
    finally:
        f.close()
    
    return commodities


def load_nodes(tntp_nodes, ftn=None):
    """Read network nodes.

    Parameters
    ----------
    tntp_nodes : str
        File path.
    ftn : int, optional
        First through node, signifies if network contains zones.

    Returns
    -------
    nodes : dict
        Nodes data.
    """
    def _read_node(line: str, ftn=None):
        n = dict()
        n['node'] = line[0].strip()
        n['x'] = float(line[1].strip())
        n['y'] = float(line[2].strip())
        n['zone'] = False
        if ftn is not None:
            try:
                ftn = int(ftn)
                node = int(n['node'])
                if node < ftn:
                    n['zone'] = True
            except:
                pass
        return n
    nodes = []

    f = dummy_file_handler(tntp_nodes)
    try:
        for line in f:
            line = line.strip()

            # skip emtpy lines
            if len(line) == 0:
                continue

            # skip header line
            if line.startswith("Node"):
                continue

            cols = clever_split(line)

            # read nodes
            if len(cols) > 1:
                nodes.append(_read_node(cols, ftn=ftn))
    finally:
        f.close()

    return nodes


def read_tntp(netfile, trips=None, nodefile=None, cost_type: str = "auto"):
    net = pd.read_csv(netfile, skiprows=8, sep='\t')
    net.columns = [s.strip().lower() for s in net.columns]
    
    # Load network edges
    st = net[["init_node", "term_node"]].values.astype(str)
    bounds = np.c_[np.full(len(st), 0), np.full(len(st), np.inf)]
    edge_data = (st, bounds)

    # Get first through node
    ftn = 1
    f = dummy_file_handler(netfile)
    try:
        for line in f:
            if line.startswith("<FIRST THRU NODE>"):
                ftn = int(line.split(">")[1].strip())
                break
    finally:
        f.close()
    
    # Build network cost
    cost_data = None
    
    def is_power_int(power: pd.Series):
        if issubclass(net.power.dtype.type, numbers.Integral):
            return True
        if np.allclose(power.astype(int), power):
            return True
        return False
    
    if (cost_type == "polynomial"
            or (cost_type == "auto" and is_power_int(net.power))):
        # If powers are integers, prepare polynomial cost
        net.power = net.power.astype(int)
        coeffs = np.zeros((len(net), max(net.power) + 1), dtype=float)
        fac = net.b / (net.capacity**net.power) * net.free_flow_time
        coeffs[:, 0] = net.free_flow_time.values
        for i, (p, fac) in enumerate(list(zip(net.power, fac))):
            coeffs[i, p] = fac
        cost_data = coeffs
    
    if cost_type == "symbolic":
        # Symbolic Cost with F = LINK_TRAVEL_TIME
        # Prepare coefficients for symbolic cost
        coeffs = {att: net[att].values for att in ["capacity", "free_flow_time", "b", "power"]}
        kw = {"F": LINK_TRAVEL_TIME}
        cost_data = (coeffs, kw)
    
    # Build or load nodes
    if nodefile is not None:
        node = pd.DataFrame(load_nodes(nodefile))
    else:
        nodes = np.sort(pd.Series(st.ravel().astype(int)).unique())
        node = pd.DataFrame({"node": nodes})
    node.node = node.node.astype(int)
    zone = (node.node < ftn).values
    labels = node.node.values.astype(str)
    xy = None
    if "x" in node.columns and "y" in node.columns:
        xy = node[["x", "y"]].values
    node_data = (labels, xy, zone)
    
    demand_data = None
    if trips is not None:
        demand_data = load_trips(trips)
    
    return edge_data, node_data, cost_data, demand_data
