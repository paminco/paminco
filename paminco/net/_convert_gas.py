import copy
import math
from lxml import etree
import numpy as np


class Quantity():
    """Class representing a (physical) quantity of some type.
    
    A quantity has a ``value`` and a ``unit`` and provides basic
    operations such as addition and multiplication with other
    quantities.
    """
    
    def __init__(self, value, unit=None):
        self._val = value
        if unit is None:
            self._unit = list(self._units.keys())[0]
        else:
            self._unit = unit.strip()
    
    @staticmethod
    def from_unit(value, unit):
        """Initialize a Quantity matching the units given in ``unit``."""
        if unit is None:
            return value
        try:
            Q = class_from_unit(unit)
            if type(Q) != str:
                return Q(value, unit)
        except TypeError:
            pass
        unit_split = unit.split('_per_')
        if len(unit_split) < 2:
            raise ValueError(f"Could not parse unit {str(unit)}")
        return MixedQuantity(value, [unit_split[0]], unit_split[1:])
    
    def to(self, unit):
        """Convert this quantity to ``unit``."""
        new_val = self.value(unit)
        self._val = new_val
        self._unit = unit
        return self
        
    def value(self, unit=None):
        """Return the value of this quantity, optionally with a specific unit"""
        if unit is None:
            unit = list(self._units.keys())[0]
        else:
            unit = unit.strip()
        if unit == self._unit:
            return self._val
        f1, o1 = self._unwrap_unit(self._unit)
        f2, o2 = self._unwrap_unit(unit)
        return (self._val * f1 - o1) / f2 + o2
      
    def _unwrap_unit(self, unit):
        unit = self._units[unit]
        if isinstance(unit, tuple):
            return unit
        return (unit, 0)
        
    def __add__(self, other, sgn=1):
        if isinstance(other, Quantity):
            if self._type != other._type:
                raise TypeError("Cannot add " + self._type + " and " + other._type)
            v = self.value(self._unit) + sgn * other.value(self._unit)
        else:
            v = self.value(self._unit) + sgn * float(other)
        return self.__class__(v, self._unit)
    
    def __radd__(self, other, sgn=1):
        v = self.value(self._unit) + sgn * float(other)
        return self.__class__(v, self._unit)
    
    def __mul__(self, other):
        if isinstance(other, Quantity):
            return MixedQuantity(self._val * other._val, [self._unit, other._unit])
        return self.__class__(self._val * float(other), self._unit)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            return self._val / other._val
        if isinstance(other, Quantity):
            return MixedQuantity(self._val / other._val, [self._unit], [other._unit])
        return self.__class__(self._val / float(other), self._unit)
    
    def __sub__(self, other):
        return self.__add__(other, sgn=-1)
    
    def __rsub__(self, other):
        return self.__add__(other, sgn=-1)
    
    def __pow__(self, other):
        if not isinstance(other, int):
            raise TypeError("Non-integer powers of Quantities are not allowed.")
        return MixedQuantity(self._val ** other, [self._unit] * other, [])
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('== not supported between ' + self._type + ' and ' + str(other.__class__))
        v2 = other.value(self._unit)
        return v2 == self._val
    
    def __gt__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('== not supported between ' + self._type + ' and ' + str(other.__class__))
        v2 = other.value(self._unit)
        return self._val > v2
    
    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('== not supported between ' + self._type + ' and ' + str(other.__class__))
        v2 = other.value(self._unit)
        return self._val < v2
    
    def __str__(self):
        return "{:s} {:.1f} {:s}".format(self._type, self._val, self._unit)

    
class MixedQuantity():
    """A class representing a mixed quantity.
    
    A mixed quantity is a (physical) quantity that is made of more than
    one basic unit. The class provides basic arithmetic operations
    and tools to convert units.
    """
    
    def __init__(self, value, units_n=None, units_d=None, no_cancel=False):
        self._val = value
        self._un = units_n if units_n is not None else []
        self._ud = units_d if units_d is not None else []
        if not no_cancel:
            self._clean_units()
        
    def _clean_units(self):
        quant_n = [class_from_unit(x) for x in self._un]
        quant_d = [class_from_unit(x) for x in self._ud]
        del_n = []
        del_d = []
        new_val = self._val
        for i, Q in enumerate(quant_n):
            k = -1
            for j, Q2 in enumerate(quant_d):
                if Q == Q2 and j not in del_d:
                    k = j
                    break
            if not k == -1:
                new_val = Q(new_val, self._un[i]).value(self._ud[k])
                del_n += [i]
                del_d += [k]
        self._un = [u for i, u in enumerate(self._un) if i not in del_n]
        self._ud = [u for i, u in enumerate(self._ud) if i not in del_d]
        self._val = new_val
        
    def value(self, units_n=None, units_d=None):
        """Return the value of this quantity.
        
        Parameters
        ----------
        units_n : str, default=None
            The unit(s) for the numerator, if necessary the value
            is converted to these units
        units_d : str, default=None
            The unit(s) for the denominator, if necessary the value
            is converted to these units
        """
        v = self._val
        if units_n is not None:
            v = MixedQuantity._conv(v, self._un, units_n)
        if units_d is not None and v != 0:
            v = 1 / MixedQuantity._conv(1 / v, self._ud, units_d)
        return v
    
    def _conv(val, units1, units2):
        units2 = copy.copy(units2)
        for u1 in units1:
            for j, u2 in enumerate(units2):
                if u1 == u2:
                    del units2[j]
                    break
                Q1 = class_from_unit(u1)
                Q2 = class_from_unit(u2)
                if Q1 == Q2:
                    x = Q1(val, u1)
                    val = x.value(u2)
                    del units2[j]
                    break
        if len(units2) != 0:
            raise TypeError("Units do not match!")
        return val

    def __add__(self, other, sgn=1):
        if isinstance(other, MixedQuantity):
            val2 = other.value(self._un, self._ud)
            return MixedQuantity(self._val + sgn * val2, self._un, self._ud)
        val2 = float(other)
        return MixedQuantity(self._val + sgn * val2, self._un, self._ud)
    
    def __sub__(self, other):
        return self.__add__(other, -1)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, MixedQuantity):
            return MixedQuantity(self._val * other._val, self._un + other._un, self._ud + other._ud)
        if isinstance(other, Quantity):
            return MixedQuantity(self._val * other._val, self._un + [other._unit], self._ud)
        return MixedQuantity(self._val * float(other), self._un, self._ud)
    
    def __truediv__(self, other):
        if isinstance(other, MixedQuantity):
            un1 = copy.copy(self._un)
            ud1 = copy.copy(self._ud)
            un2 = copy.copy(other._un)
            ud2 = copy.copy(other._ud)
            for x in other._un:
                if x in un1:
                    un1.remove(x)
                    un2.remove(x)
            for x in other._ud:
                if x in ud1:
                    ud1.remove(x)
                    ud2.remove(x)
            new_un = un1 + ud2
            new_ud = ud1 + un2
            if len(new_un) + len(new_ud) == 0:
                return self._val / other._val
            return MixedQuantity(self._val / other._val, new_un, new_ud)
        if isinstance(other, Quantity):
            return MixedQuantity(self._val / other._val, self._un, self._ud + [other._unit])
        return MixedQuantity(self._val / float(other), self._un, self._ud)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __str__(self, displaystyle=False):
        out = "MIXED "
        num_str = " ".join(self._un)
        if len(self._un) == 0:
            num_str = '1'
        den_str = " ".join(self._ud)
        if len(self._ud) == 0:
            den_str = '1'
        if not displaystyle:
            out += str(self._val) + ' ' + num_str + ' / ' + den_str
            return out
        un_len = max(len(num_str), len(den_str))
        val_str = str(self._val)
        val_len = len(val_str)
        out = (" " * (val_len + 1)) + ("{:^" + str(un_len) + "s}").format(num_str) + '\n'
        out += val_str + ' ' + ("-" * un_len) + '\n'
        out += (" " * (val_len + 1)) + ("{:^" + str(un_len) + "s}").format(den_str)
        return out


class Time(Quantity):
    """Quantity time with according units."""
    
    _type = "TIME"
    _units = {'s': 1, 'ms': 1 / 1000, 'min': 60, 'minute': 60,
              'minutes': 60, 'h': 3600, 'hour': 3600, 'hours': 3600}


class Area(Quantity):
    """Quantity area with according units."""

    _type = "AREA"
    _units = {'m_square': 1, 'm^2': 1}
    
    
class Pressure(Quantity):
    """Quantity pressure with according units."""

    _type = "PRESSURE"
    _units = {'Pa': 1, 'hPa': 100, 'bar': 1e5}


class Temperature(Quantity):
    """Quantity temperature with according units."""

    _type = "TEMERATURE"
    _units = {'K': 1, 'Kelvin': 1, 'Celsius': (1, -273.15)}


class Length(Quantity):
    """Quantity length with according units."""

    _type = "LENGTH"
    _units = {'m': 1, 'mm': 1 / 1000, 'cm': 1 / 100, 'dm': 1 / 10, 'km': 1000, 'meter': 1}


class Weight(Quantity):
    """Quantity weight with according units."""
    
    _type = "WEIGHT"
    _units = {'kg': 1, 'Grams': 1 / 1000, 'g': 1 / 1000}


class Volume(Quantity):
    """Quantity volume with according units."""

    _type = "VOLUME"
    _units = {'m^3': 1, 'm_cube': 1, 'liters': 1 / 1000, '1000m_cube': 1000}


class Energy(Quantity):
    """Quantity energy with according units."""

    _type = "JOULE"
    _units = {'J': 1, 'kJ': 1000, 'MJ': 1e6}
    

class AmountOfSubstance(Quantity):
    """Quantity amount of substance with according units."""

    _type = "AMOUNT OF SUBSTANCE"
    _units = {'mol': 1, 'kmol': 1e3, 'Mmol': 1e6, '': 1 / (6.02214076 * 1e23)}


QUANTITY_TYPES = [Time, Pressure, Temperature, Length, Area, Weight, Volume, Energy, AmountOfSubstance]
MIXED_UNITS = {'W': (['J'], ['s'])}


def class_from_unit(unit):
    unit = unit.strip()
    for Q in QUANTITY_TYPES:
        if unit in Q._units.keys():
            return Q
    try:
        un, ud = MIXED_UNITS[unit]
        return ''.join(un) + '_per_' + '_per_'.join(ud)
    except KeyError:
        pass
        
    raise TypeError('Unknown Unit ' + str(unit))


GAS_CONSTANT = MixedQuantity(8.314462618, ['J'], ['mol', 'K'])


def indent(elem, level=0):
    """Helper function for xml indentation."""
    i = "\n" + level * "  "
    j = "\n" + (level - 1) * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            indent(subelem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem


def remove_namespace(tag, ns):
    """Remove namespace from xml tag."""
    for n in ns.values():
        tag = tag.replace('{' + n + '}', '')
    return tag


def gaslib_to_network_data(network_file, scenario_file, contract_aux_elements=True, debug=False):
    """Read a gaslib instance from files and create network data.

    The function returns data that can be passed immediatley to the
    constructor of Network.
    
    Parameters
    ----------
    network_file : str
        The filename of the gaslib XML network file
    scenario_file : str
        The filename of the gaslib XML scenario file
        to be used as demands
    contract_aux_elements : boolean, default=True
        If set to True, auxiliary components (i.e. componentss that are not
        pipes, such as vales and compressors) are contracted. If set to False
        these components are modeled as pipes with very small resistances
    debug : boolean, default=False
        If set to True, some debug information are printed

    Returns
    -------
    edge_data : tuple
        The network edge data
    node_data : tuple
        The network node data
    cost_data : tuple
        The network polynomial cost data
    demand_data : tuple
        The network demand data
    """

    # Read the demand from the scenario file
    scn_tree = etree.parse(scenario_file)
    scn_root = scn_tree.getroot()
    
    demand = read_gaslib_scn(scn_root)

    # Read the network XML file
    net_tree = etree.parse(network_file)
    root = net_tree.getroot()
    
    nodes = read_gaslib_nodes(root)
    
    # Compute average node values for node properties
    # Needed because some "global" properties such as
    # gas temperature are needed in the computation of the beta
    # coefficients. The average value of all nodes are used for
    # these computations
    avg_n_vals = dict()
    for n in nodes:
        for p in nodes[n]['p']:
            avg_n_vals[p] = avg_n_vals.get(p, []) + [nodes[n]['p'][p]]
    
    for p in avg_n_vals:
        vals = avg_n_vals.get(p, [])
        if len(vals) != 0:
            avg_n_vals[p] = sum(vals) / len(vals)
            
    if debug:
        print("\n== GLOBAL NODE VALUES (AVG) ==")
        for p in avg_n_vals:
            print("{:30s}: {:s}".format(str(p), str(avg_n_vals[p])))
    
    # Read pipe data from the XML file
    edges = read_gaslib_pipes(root)
    # Compute the minimal beta of all edges
    # (used as beta for auxiliary elements, if contract_aux_elements == False)
    min_beta = math.inf
    for e in edges:
        edges[e]['beta'] = calc_beta(edges[e], nodes, avg_n_vals)
        min_beta = min(min_beta, edges[e]['beta'])

    if debug:
        print('MIN BETA', min_beta)

    # Read further components (non-pipe edges) from XML
    aux_elements = read_gaslib_aux_elements(root)
    
    # Handle the auxiliary components
    contract_gaslib_instance(nodes, edges, demand, aux_elements, contract_aux_elements, min_beta, debug)

    return raw_data_to_network_data(nodes, edges, demand)


def raw_data_to_network_data(nodes, edges, demand):
    """Convert the raw data of gaslib instances to network data."""
    node_data = (
        np.array([n for n in nodes]),
        np.array([[nodes[n]['x'], nodes[n]['y']] for n in nodes])
    )
    edge_data = (
        [[edges[e]['from'], edges[e]['to']] for e in edges],
        np.array([[-np.inf, np.inf] for _ in edges])
    )
    cost_data = (
        np.array([[0, 0, edges[e]['beta']] for e in edges]),
        [True for _ in edges]
    )
    demand_data = {
        n: demand[n]['b']
        for n in demand
    }

    return edge_data, node_data, cost_data, demand_data


def read_gaslib_scn(xml_root):
    """Read raw scenario data from gaslib xml."""
    demand = dict()
    scn_node = xml_root.find("scenario", xml_root.nsmap)
    for node in scn_node.findall("node", xml_root.nsmap):
        for flow_node in node.findall("flow", xml_root.nsmap):
            if flow_node.attrib['bound'] == 'both':
                lb = ub = float(flow_node.attrib['value'])
            elif flow_node.attrib['bound'] == 'lower':
                lb = float(flow_node.attrib['value'])
            elif flow_node.attrib['bound'] == 'upper':
                ub = float(flow_node.attrib['value'])
        n = {'type': node.attrib['type'], 'lb': lb, 'ub': ub}
        demand[node.attrib['id']] = n
        
    # Find satisfying nomination
    for n in demand:
        if demand[n]['type'] == "entry":
            demand[n]['b'] = - demand[n]['lb']
        if demand[n]['type'] == "exit":
            demand[n]['b'] = demand[n]['lb']
    return demand
    
    
def read_gaslib_nodes(xml_root):
    """Read raw node data from gaslib xml."""
    nodes = dict()
    
    # READ NODES
    nodes_node = xml_root.find("framework:nodes", xml_root.nsmap)
    for x in nodes_node:
        prop = dict()
        for p in x:
            pname = remove_namespace(p.tag, xml_root.nsmap)
            val = float(p.attrib.get('value', 0))
            unit = p.get('unit', None)
            prop[pname] = Quantity.from_unit(val, unit)
        x
        x_pos = x.attrib['x']
        y_pos = x.attrib['y']
        nodes[x.attrib['id']] = {'x': x_pos, 'y': y_pos, 'p': prop}
        
    return nodes


def read_gaslib_pipes(xml_root):
    """Read raw pipe data from gaslib xml."""
    edges = dict()
    
    edges_node = xml_root.find("framework:connections", xml_root.nsmap)
    for pipe in edges_node.findall("pipe", xml_root.nsmap):
        prop = dict()
        for p in pipe:
            if not isinstance(p, etree._Comment):
                pname = remove_namespace(p.tag, xml_root.nsmap)
                val = float(p.attrib.get('value', 0))
                unit = p.get('unit', None)
                prop[pname] = Quantity.from_unit(val, unit)
        v = pipe.attrib['from']
        w = pipe.attrib['to']
        edges[pipe.attrib['id']] = {'from': v, 'to': w, 'p': prop}
    
    return edges


def read_gaslib_aux_elements(xml_root):
    """Read raw auxiliary edge data from gaslib xml."""
    aux_elements = dict()
    
    edges_node = xml_root.find("framework:connections", xml_root.nsmap)
    
    compressors = edges_node.findall("compressorStation", xml_root.nsmap)
    valves = edges_node.findall("valve", xml_root.nsmap)
    shortpipes = edges_node.findall("shortPipe", xml_root.nsmap)
    controlvalves = edges_node.findall("controlValve", xml_root.nsmap)
    resistors = edges_node.findall("resistor", xml_root.nsmap)
    
    all_elements = compressors + valves + shortpipes + controlvalves + resistors
    
    for el in all_elements:
        prop = dict()
        for p in el:
            pname = remove_namespace(p.tag, xml_root.nsmap)
            prop[pname] = Quantity.from_unit(float(p.attrib.get('value', 0)), p.get('unit', None))
        aux_elements[el.attrib['id']] = {'from': el.attrib['from'], 'to': el.attrib['to'], 'p': prop}
    
    return aux_elements


def contract_gaslib_instance(nodes, edges, demand, aux_elements, contract_aux_edges, min_beta, debug=False):
    """Contract all auxiliary edge elements (if contract_aux_edge == False, create low resistance edges instead)."""
    # Dict containg the node ids of contracted nodes
    # contracted_nodes['w'] = 'v' means that node v and w are contracted,
    # where node 'w' will be deleted
    contracted_nodes = dict()
    if contract_aux_edges:
        # Contract all auxillary elements like compressors and valves
        for ae in aux_elements:
            v, w = (aux_elements[ae]['from'], aux_elements[ae]['to'])
            while v in contracted_nodes and not v == contracted_nodes[v]:
                v = contracted_nodes[v]
            if w in contracted_nodes:
                if debug:
                    print(w, "was already contracted", w, '->', v)
                u = contracted_nodes[w]
                for x in contracted_nodes:
                    if contracted_nodes[x] == u:
                        if debug:
                            print("Updating ", x, ':', x, '->', u, ' updated to ', x, '->', v)
                        contracted_nodes[x] = v
                if debug:
                    print("Contracting also", u, '->', v)
                contracted_nodes[u] = v
            for x in contracted_nodes:
                if contracted_nodes[x] == w:
                    if debug:
                        print("! Updating ", x, ':', x, '->', w, ' updated to ', x, '->', v)
                    contracted_nodes[x] = v
            contracted_nodes[w] = v
    else:
        # Emulate zero resistance pipes for all auxillary elements (like compressors, valves)
        for ae in aux_elements:
            aux_elements[ae]['beta'] = min_beta * 1e-10
        edges.update(aux_elements)
    
    if debug:
        print("== CONTRACTED NODES ==")
        for v in contracted_nodes:
            print("Contracting node", v, "into node", contracted_nodes[v])
    # Update Nodes with contracted nodes (if necessary)
    for w in contracted_nodes:
        v = contracted_nodes[w]
        if v != w:
            del nodes[w]
            if w in demand:
                if v in demand:
                    demand[w]['b'] += demand[v]['b']
                    if debug:
                        print("Contracting two entries/exits", v, 'and', w, "\nNew b ", demand[w]['b'])
                demand[v] = demand[w]
                del demand[w]
    # Update Edges by updating node ids
    for eid in edges:
        e = edges[eid]
        if e['from'] in contracted_nodes:
            e['from'] = contracted_nodes[e['from']]
        if e['to'] in contracted_nodes:
            e['to'] = contracted_nodes[e['to']]


def calc_beta(edge, nodes, avg_n_vals):
    r"""Compute the resistance coefficient beta for the pipe edge.
    
    Compute the resistance value :math:`\beta_e` for the edge
    given by the raw edge data in ``edge`` with the formula

    .. math::
        \beta_e = \bigg(\frac{4}{\pi}\bigg)^2 \frac{L_e}{D^5_e}
        \frac{R}{m} z(p_e, T) T \lambda_e
    where

    .. math::
        \lambda_e = \bigg( 2 \log_10 \bigg(\frac{D_e}{k_e}\bigg)
        +1.138 \bigg)^2
    and

    .. math::
        z(p_e, T) = 1 + 0.257 \frac{p_e}{p_c} -
        0.533 \frac{\frac{p_e}{p_c}}{\frac{T}{T_c}}
    
    with

    .. math::
        p_e = \frac{\min (\underline{p}_u, \underline{p}_v)
        + \max (\bar{p}_u, \bar{p}_v)}{2}

    for the edge :math:`e = (u, v)` with the constants

    * :math:`L_e` - the length of the pipe
    * :math:`D_e` - the diameter of the pipe
    * :math:`R` - the universal gas constant
    * :math:`m` - the molar mass of the gas
    * :math:`T` - the temperature of the gas
    * :math:`k_e` - the roughness of the pipe
    * :math:`z(p_e, T)` - the compressibility factor of the gas
    * :math:`p_c` - the pseudo-critical pressure of the gas
    * :math:`T_c` - the pseudo-critical temperature of the gas
    """
    u = nodes[edge['from']]
    v = nodes[edge['to']]
    # Compute pa
    pl = min(u['p']['pressureMin'], v['p']['pressureMin'])
    pu = max(u['p']['pressureMax'], v['p']['pressureMax'])
    pa = ((pl + pu) / 2)
    pa.to('Pa')

    # Compute z_(p_e, T)
    pc = avg_n_vals['pseudocriticalPressure'].to('Pa')
    prq = pa / pc
    T = avg_n_vals['gasTemperature'].to('K')
    Tc = avg_n_vals['pseudocriticalTemperature'].to('K')
    teq = T / Tc
    z = 1 + 0.257 * prq - 0.533 * prq / teq

    da = edge['p']['diameter']
    ka = edge['p']['roughness']
    lambdaa = 1 / ((2 * math.log10(da / ka) + 1.138) ** 2)

    beta = GAS_CONSTANT / avg_n_vals['molarMass']
    beta = beta * z * T * lambdaa
    da5 = (da.to('m')) ** 5

    beta = (4 / math.pi) ** 2 * beta * edge['p']['length'] / da5
    
    # Transform units of beta
    fac0 = MixedQuantity(1, ['kg', 'm', 'm'], ['s', 's', 'J'])
    fac1 = MixedQuantity(1 / 100000, ['bar', 'm', 's', 's'], ['kg'])
    fac2 = MixedQuantity(3600, ['s'], ['h'], no_cancel=True)
    
    beta = beta * fac0 * fac1 * fac1 / fac2 / fac2
    beta = beta * avg_n_vals['normDensity'] * avg_n_vals['normDensity']
    
    return beta.value() * 1e6
