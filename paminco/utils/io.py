"""Module that provides utils for loading objects via pickle, XML."""
from xml.etree import ElementTree
from xml.dom import minidom
import pickle


def save_object(obj, file: str) -> None:
    """Save object via pickle.
    
    Parameters
    ----------
    obj : all pickable types
        Object to be saved.
    file : str
        Path to save object to.
    
    See Also
    --------
    pickle.load
    """
    filehandler = open(file, 'wb')
    pickle.dump(obj, filehandler)


def load_object(file):
    """Load object via pickle.

    Parameters
    ----------
    file :
        Read file from pickled data in `file`.
    
    Returns
    -------
    obj
        Loaded object.
    
    See Also
    --------
    pickle.load
    """
    if isinstance(file, str):
        file = open(file, 'rb')
    return pickle.load(file)


def prettify_xml(file) -> None:
    """Prettify an xml file.

    Use mainly to debug as whitespace is significant in XML format.

    Parameters
    ----------
    file : str
        File to prettify.

    """
    def prettify(elem):
        """Return a pretty-printed XML string for the Element.

        https://stackoverflow.com/questions/17402323 @Maxime Chéramy.
        """
        rough_string = ElementTree.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    tree = ElementTree.parse(file)
    root = tree.getroot()

    with open(file, 'w') as f:
        f.write(prettify(root))
