import copy

import numpy as np

class check():
    def __init__(self):
        '''Functions that are used to validate multiple things. These should be formatted to give useful error messages when interfacing eith the quantum computer.'''

        pass


    @staticmethod
    def check_type(obj: object, type_to_check):
        '''Checks the type of the object'''
        error_header = "\n----------------------------------------\nQuantum Computer Simulator Error\n----------------------------------------\n"
        assert (type(obj) == type_to_check), f"{error_header}An object has an unexpected type. The object \n{obj}\nhas type {type(obj)}. The type expected was {type_to_check}."

    @staticmethod
    def check_in_list(obj: object, list_to_check: list):
        '''Checks if something is present in a list'''
        error_header = "\n----------------------------------------\nQuantum Computer Simulator Error\n----------------------------------------\n"
        assert (obj in list_to_check), f"{error_header}An object was not found in a list. The object \n{obj}\nis not in the following list:\n{list_to_check}."

    @staticmethod
    def check_array_shape(obj: np.array, shape_to_check: tuple):
        error_header = "\n----------------------------------------\nQuantum Computer Simulator Error\n----------------------------------------\n"
        assert np.all(obj.shape == shape_to_check), f"{error_header}An object does not have the expected numpy array shape. When checking the shape of the object \n{obj}\nit was found to have the shape {obj.shape}. Instead, it should have the shape {shape_to_check}."

    @staticmethod
    def check_array_length(obj: np.array, length_to_check: int):
        error_header = "\n----------------------------------------\nQuantum Computer Simulator Error\n----------------------------------------\n"
        assert (len(obj) == length_to_check), f"{error_header}An object does not have the expected length. When checking the length of the object \n{obj}\nit was found to have a length of {len(obj)}. Instead, it should have a length of {length_to_check}."