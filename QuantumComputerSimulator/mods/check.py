import numpy as np

class check():

    error_header = "\n----------------------------------------\nQuantum Computer Simulator Error\n----------------------------------------\n"

    def __init__(self):
        '''
        Custom error class mainly for handling when the user incorrectly interfaces with the package.

        These Functions are used to validate multiple things. These should be formatted to give useful error messages when interfacing with the quantum computer. Will also print the error producing object in addition to what it was being compared to.

        Can be used for debugging.
        '''
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
        '''Checks if a numpy array has the correct shape.'''
        assert np.all(obj.shape == shape_to_check), f"{cls.error_header}An object does not have the expected numpy array shape. When checking the shape of the object \n{obj}\nit was found to have the shape {obj.shape}. Instead, it should have the shape {shape_to_check}."

    @classmethod
    def check_array_length(cls, obj: list, length_to_check: int):
        '''Checks if a python list has the correct length.'''
        assert (len(obj) == length_to_check), f"{cls.error_header}An object does not have the expected length. When checking the length of the object \n{obj}\nit was found to have a length of {len(obj)}. Instead, it should have a length of {length_to_check}."

    @classmethod
    def check_sum(cls, obj: list, check_sum: float):
        '''Checks if the elements of a list sum to `check_sum`. Useufl for normalisation checks.'''
        sum_of_object = sum(obj)
        assert (np.around(sum_of_object, decimals = 2) == check_sum), f"{cls.error_header}The elements of the object\n{obj}\ndo not sum to {check_sum}. Instead, it sums to {sum_of_object}."