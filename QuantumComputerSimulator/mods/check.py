import copy

import numpy as np

class check():

    error_header = "\n----------------------------------------\nQuantum Computer Simulator Error\n----------------------------------------\n"

    def __init__(self):
        '''Functions that are used to validate multiple things. These should be formatted to give useful error messages when interfacing eith the quantum computer.'''
        pass

    @classmethod
    def check_type(cls, obj: object, type_to_check):
        '''Checks the type of the object.'''
        assert (type(obj) == type_to_check), f"{cls.error_header}An object has an unexpected type. The object \n{obj}\nhas type {type(obj)}. The type expected was {type_to_check}."

    @classmethod
    def check_in_list(cls, obj: object, list_to_check: list):
        '''Checks if something is present in a list'''
        assert (obj in list_to_check), f"{cls.error_header}An object was not found in a list. The object \n{obj}\nis not in the following list:\n{list_to_check}."

    @classmethod
    def check_array_shape(cls, obj: np.array, shape_to_check: tuple):
        assert np.all(obj.shape == shape_to_check), f"{cls.error_header}An object does not have the expected numpy array shape. When checking the shape of the object \n{obj}\nit was found to have the shape {obj.shape}. Instead, it should have the shape {shape_to_check}."

    @classmethod
    def check_array_length(cls, obj: np.array, length_to_check: int):
        assert (len(obj) == length_to_check), f"{cls.error_header}An object does not have the expected length. When checking the length of the object \n{obj}\nit was found to have a length of {len(obj)}. Instead, it should have a length of {length_to_check}."