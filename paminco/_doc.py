def from_xml():
    """Load object from ``.xml`` data.

    Parameters
    ----------
    data : str, file, ElementTree or Element
        Initialize object from xml file by passing
            * filename as str,
            * file object that contains XML data,
            * the XML ElementTree,
            * or the root Element of the ElementTree.
    kwargs : keyword arguments, optional
        Keyword arguments for constructing object.

    See Also
    --------
    xml.etree.ElementTree
    xml.etree.ElementTree.parse
    """
    pass  # use for uniform docstrings


def from_xml_shared():
    """Load object from ``.xml`` data

    Parameters
    ----------
    shared : Shared
        Shared object for network classes.
    data : str, file, ElementTree or Element
        Initialize object from xml file by passing
            * filename as str,
            * file object that contains XML data,
            * the XML ElementTree,
            * or the root Element of the ElementTree.
    kwargs : keyword arguments, optional
        Keyword arguments for constructing object.

    See Also
    --------
    xml.etree.ElementTree
    xml.etree.ElementTree.parse
    """
    pass  # use for uniform docstrings


def make_save_dict():
    """Generate dict to save object with numpy.savez.

    Parameters
    ----------
    prefix : str, default=""
        Save attributes with ``names = prefix + internal_name``.
    save_dict : dict, optional
        If given, dict elements will be added to dict, else a new
        dict is created.

    Returns
    -------
    save_dict : dict
        Dictionary to be saved with :func:`~numpy.savez`.

    See Also
    --------
    numpy.savez
    save_to_numpy
    """
    pass  # use for uniform docstrings


def save_to_numpy():
    """Save object into a single file in uncompressed ``.npz`` format.

    Parameters
    ----------
    file : str or file
        Filename as str or open file.
    kwargs : keyword arguments, optional
        Keyword arguments saved to file.

    See Also
    --------
    make_save_dict
    numpy.savez
    numpy.lib.npyio.NpzFile
    """
    pass  # use for uniform docstrings


def from_npz():
    """Construct object from ``.npz`` file.

    Parameters
    ----------
    data : str or NpzFile
        Filename as str or NpzFile.
    prefix : str, default=""
        Object data is stored with ``key = (prefix + internal_name)``.

    Returns
    -------
    obj
        Object.

    See Also
    --------
    numpy.lib.npyio.NpzFile
    """
    pass  # use for uniform docstrings


def from_npz_shared():
    """Construct object from ``.npz`` file.

    Parameters
    ----------
    shared : Shared
        Shared object for network classes.
    data : str or NpzFile
        Filename as str or NpzFile.
    prefix : str, default=""
        Object data is stored with ``key = (prefix + internal_name)``.

    Returns
    -------
    obj
        Object.

    See Also
    --------
    numpy.lib.npyio.NpzFile
    """
    pass  # use for uniform docstrings


def reset_cache(self, hard: bool = False) -> None:
    """Reset cache values.

    Parameters
    ----------
    hard : bool, default=False
        If ``False``, valid flag of cached elements is set to False.
        If ``True``, cached elements will be set to None.
    """
    pass
