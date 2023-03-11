import copy

import numpy as np

class check():
    def __init__(self):
        '''Functions that are used to validate multiple things. These should be formatted to give useful error messages when interfacing eith the quantum computer.'''
        pass

    @staticmethod
    def check_type(obj: object, type_to_check):
        '''Checks the type of the object'''
        assert (type(obj) == type_to_check), f"Type of {obj} was found to be {type(obj)}. Instead, the type should be {type_to_check}"

    @staticmethod
    def check_in_list(obj: object, list_to_check: list):
        assert (obj in list_to_check), f"The object \n{obj}\n is not in the following list:\n{list_to_check}."

    @staticmethod
    def check_array_shape(obj: np.array, shape_to_check: tuple):
        assert np.all(obj.shape == shape_to_check), f"When checking the shape of \n{obj}\n It was found to be {obj.shape}. Instead, it should be \n{shape_to_check}."
