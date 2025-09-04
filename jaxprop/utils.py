import re
import numbers
import numpy as np

from collections.abc import Iterable

import numpy as np


def is_float(element: any) -> bool:
    """
    Check if the given element can be converted to a float.

    Parameters
    ----------
    element : any
        The element to be checked.

    Returns
    -------
    bool
        True if the element can be converted to a float, False otherwise.
    """

    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def is_numeric(value):
    """
    Check if a value is a numeric type, including both Python and NumPy numeric types.

    This function checks if the given value is a numeric type (int, float, complex)
    in the Python standard library or NumPy, while explicitly excluding boolean types.

    Parameters
    ----------
    value : any type
        The value to be checked for being a numeric type.

    Returns
    -------
    bool
        Returns True if the value is a numeric type (excluding booleans),
        otherwise False.
    """
    # Exclude Python bool
    if isinstance(value, bool):
        return False

    # Python numbers (int, float, complex)
    if isinstance(value, numbers.Number):
        return True

    # NumPy scalar types
    if isinstance(value, np.generic):
        return np.issubdtype(type(value), np.number) and not np.issubdtype(type(value), np.bool_)

    # NumPy arrays
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.number) and not np.issubdtype(value.dtype, np.bool_)

    return False


def check_lists_match(list1, list2):
    """
    Check if two lists contain the exact same elements, regardless of their order.

    Parameters
    ----------
    list1 : list
        The first list for comparison.
    list2 : list
        The second list for comparison.

    Returns
    -------
    bool
        Returns True if the lists contain the exact same elements, regardless of their order.
        Returns False otherwise.

    Examples
    --------
    >>> check_lists_match([1, 2, 3], [3, 2, 1])
    True

    >>> check_lists_match([1, 2, 3], [4, 5, 6])
    False

    """
    # Convert the lists to sets to ignore order
    list1_set = set(list1)
    list2_set = set(list2)

    # Check if the sets are equal (contain the same elements)
    return list1_set == list2_set



def print_dict(data, indent=0, return_output=False):
    """
    Recursively prints nested dictionaries with indentation or returns the formatted string.

    Parameters
    ----------
    data : dict
        The dictionary to print.
    indent : int, optional
        The initial level of indentation for the keys of the dictionary, by default 0.
    return_output : bool, optional
        If True, returns the formatted string instead of printing it.

    Returns
    -------
    str or None
        The formatted string representation of the dictionary if return_output is True, otherwise None.
    """
    lines = []
    for key, value in data.items():
        line = "    " * indent + str(key) + ": "
        if isinstance(value, dict):
            if value:
                lines.append(line)
                lines.append(print_dict(value, indent + 1, True))
            else:
                lines.append(line + "{}")
        else:
            lines.append(line + str(value))

    output = "\n".join(lines)
    if return_output:
        return output
    else:
        print(output)


def print_object(obj):
    """
    Prints all attributes and methods of an object, sorted alphabetically.

    - Methods are identified as callable and printed with the prefix 'Method: '.
    - Attributes are identified as non-callable and printed with the prefix 'Attribute: '.

    Parameters
    ----------
    obj : object
        The object whose attributes and methods are to be printed.

    Returns
    -------
    None
        This function does not return any value. It prints the attributes and methods of the given object.

    """
    # Retrieve all attributes and methods
    attributes = dir(obj)

    # Sort them alphabetically, case-insensitive
    sorted_attributes = sorted(attributes, key=lambda x: x.lower())

    # Iterate over sorted attributes
    for attr in sorted_attributes:
        # Retrieve the attribute or method from the object
        attribute_or_method = getattr(obj, attr)

        # Check if it is callable (method) or not (attribute)
        if callable(attribute_or_method):
            print(f"Method: {attr}")
        else:
            print(f"Attribute: {attr}")

