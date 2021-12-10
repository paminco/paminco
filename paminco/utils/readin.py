"""Module that handles readin of XML files for all network objects."""
import re
import tempfile
import xml.etree.ElementTree as et


DEFAULT_ROOT_TAG = "network"


class FileLineIterator:
    
    def __init__(
            self,
            file,
            is_string: bool = False,
            split_lines: str = "\n",
            ):
        self.file = file
        if is_string is True:
            self.lines = file.split(split_lines)
            self.has_file = False
        else:
            if not hasattr(file, "read"):
                file = open(file, 'r')
            self.lines = file.readlines()
            self.has_file = True
            
    def close_file(self) -> None:
        if self.has_file is True:
            self.file.close()
    
    def __getitem__(self, idx):
        return self.lines.__getitem__(idx)
    
    def dump(
            self,
            file=None,
            prepend: str = "",
            append: str = "",
            tmp_suffix: str = ".txt",
            write_mode: str = "a"):
        return_file = False
        if file is None:
            # Create tempfile
            file = tempfile.mkstemp(suffix=tmp_suffix)[1]
            return_file = True
        
        if not hasattr(file, "write"):
            file = open(file, write_mode)
        
        # Add stuff to beginning of file
        file.write(prepend)
        
        # Add all lines to file
        if self.has_file is True:
            file.writelines([line for line in self])
        else:
            file.writelines([line + '\n' for line in self])
            
        # Add stuff to end of file
        file.write(append)
        
        # Return open tempfile
        if return_file is True:
            return file.name
        
        file.close()


def parse_polynomial(poly: str) -> list:
    """Get polynomial coefficients froms string.
    
    May not be used with other mathematical functions.
    
    Parameters
    ----------
    poly : str
        Polynomial string.
    
    Returns
    -------
    list of floats
        Polynomial coefficients [x_0, x_1, ..., x_n].
    """
    powers = re.findall("([\\- ]*[0-9\\.]*)([ ]*x\\^)([0-9]*)", poly)
    linear = re.findall("([\\- ]*[0-9\\.]*)([ ]*x)(?!\\^)", poly)
    constants = re.findall("(?<![x\\^]{1})([\\- ]*[0-9\\.]+)(?![0-9 \\.]*x)",
                           poly)
    coeff = dict()
    for co in constants:
        co = float(co.replace(' ', '').strip())
        coeff[0] = coeff.get(0, 0) + co
    
    for co, _ in linear:
        co = co.replace(' ', '').strip()
        co = 1 if len(co) == 0 else co
        co = -1 if co == '-' else co
        co = float(co)
        coeff[1] = coeff.get(1, 0) + co
    
    for co, _, pow in powers:
        co = co.replace(' ', '').strip()
        co = 1 if len(co) == 0 else co
        co = -1 if co == '-' else co
        co = float(co)
        pow = int(pow)
        coeff[pow] = coeff.get(pow, 0) + co
    
    deg = max(coeff.keys())
    out = [coeff.get(k, 0) for k in range(deg + 1)]
    
    return out


def parse_number(number_str: str, to_type=float) -> list:
    """Convert number from str to ``to_type``.

    Parameters
    ----------
    number_str : str
        Input string.
    to_type : dtype = float
        Return type of ``number_str``.

    Returns
    -------
    to_type
        Input converted ``to_type``.
    """
    if not isinstance(number_str, str):
        return number_str
    number_str = number_str.strip()
    return to_type(number_str.replace(" ", ""))


def xml_find_root(data, root_tag: str = DEFAULT_ROOT_TAG) -> et.Element:
    """Find root element in XML data.

    Parameters
    ----------
    data : str, xml.etree.ElementTree.ElementTree, or xml.etree.ElementTree.Element
        XML file, tree or root itself.

    Returns
    -------
    xml.etree.ElementTree.Element
        Root element in ``data``.

    Raises
    ------
    ValueError:
        ``data`` is xml.etree.ElementTree.Element with a tag that differs
        from `default_root_tag`.
    TypeError:
        ``data`` is not valid input type.
    """
    if isinstance(data, str):
        if data.endswith(".xml"):
            data = et.parse(data)
        else:
            data = et.fromstring(data)
    if isinstance(data, et.ElementTree):
        data = data.getroot()
    if isinstance(data, et.Element):
        if data.tag == root_tag:
            return data
        raise ValueError(
            f"Tag of element: '{data.tag}' does not match"
            f"default root tag '{root_tag}'"
        )
    raise TypeError(f"Invalid type for 'data': {type(data)}")


def xml_add_element(
        parent: et.Element,
        name: str,
        overwrite: bool = False,
        attrib=None,
        **kw
        ) -> et.Element:
    if attrib is None:
        attrib = {}
    
    # Get element
    elem = parent.find(name)
    
    if overwrite is True and elem is not None:
        # Remove if element exists
        parent.remove(elem)
        elem = None
    
    if elem is None:
        elem = et.SubElement(parent, name, attrib=attrib, **kw)
    
    for k, v in attrib.values():
        elem.attrib[k] = v
    
    return elem
