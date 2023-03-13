#THIS IS NOT GOOD CODE

import numpy as np
from inspect import signature as sg

I = lambda x,y: [x,y]
H = lambda x,y: [(x+y)/np.sqrt(2),(x-y)/np.sqrt(2)]


def TP(m1, m2):
    d = {}

    n1 = len(sg(m1).parameters)
    n2 = len(sg(m2).parameters)

    string = ['tp = lambda ']

    for i in range(1, n1 * n2 + 1):
        string.append(chr(ord('`') + i))
        if i != n1 * n2:
            string.append(',')

    string.append(': m1(')

    for i in range(0, n1):
        string.append('m2(')
        for j in range(0, n2):
            string.append(chr(ord('`') + n2 * i + j + 1))
            if j != n2 - 1:
                string.append(',')
        string.append(')')
        if i != n1 - 1:
            string.append(',')

    string.append(')')

    new_string = ''.join(string)

    exec(new_string, locals(), d)

    tp = d['tp']

    return tp

IH = TP(I,H)
IIH = TP(I,IH)

print(IIH(1,2,3,5,7,11,13,17))