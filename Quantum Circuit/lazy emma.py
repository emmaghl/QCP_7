import inspect
import numpy as np
import numpy as np
from inspect import signature as sg

import numpy as np
from inspect import signature as sg

I = lambda x, y: [x, y]
H = lambda x, y: [(x + y) / np.sqrt(2), (x - y) / np.sqrt(2)]

def TP(m1, m2):
    n1 = len(sg(m1).parameters)
    n2 = len(sg(m2).parameters)

    print(n1, n2)
    #     return lambda *args: [B[i](A[j](args[k] for k in range(len_A-1)),A[j](args[len_A+k] for k in range(len_A-1))) for i in range(len_B) for j in range(len_A)]
    return lambda *args: [m2(*[m1(*args[i:i+n1]) for i in range(j*n1, (j+1)*n1)]) for j in range(n2)]


IH = TP(I,H)
IIH = TP(I,TP(I,H))
# print(IIH(1,2,3,5,7,11,13,17))




#
# import inspect
# import numpy as np
#
# def TensProd(A, B):
#     len_A = len(inspect.signature(A).parameters)
#     len_B = len(inspect.signature(B).parameters)
#     def result_func(*args):
#         print(f"len_A: {len_A}, len_B: {len_B}")
#         print(f"args: {args}")
#         x = args[:len_A]
#         y = args[len_A:]
#         print(f"x: {x}, y: {y}")
#         result = [[sum([A(x[i:i+len_A], y[i:i+len_A])[k] * B(x[j+len_A:j+len_A+len_B], y[j+len_A:j+len_A+len_B])[k] for k in range(len_B)]) for j in range(0, len(x)-len_A+1)] for i in range(0, len(x)-len_A+1)]
#         return result
#     return result_func
#
# I = lambda x,y: [x,y]
# H = lambda x,y: [(x+y)/np.sqrt(2),(x-y)/np.sqrt(2)]
#
# IH = TensProd(I, H)
# print(IH((1, 2), (3, 5)))

# def TensProd(A, B):
#     # a =
#     # b =
#     # TP2 = lambda *args: [I[i](H[j](args[0],args[1]),H[j](args[2],args[3])) for i in range(len(I())) for j in range(len(H))]
#     # TP2 = lambda *args: [I[i](H[j](args[0],args[1], ..., args[len(H)]),H[j](args[len(H)+0],args[len(H)+1],..., args[len(H) + len(H)])) for i in range(len(I())) for j in range(len(H))]
#     len_A = len(inspect.signature(A).parameters)
#     len_B = len(inspect.signature(B).parameters)
#     return lambda *args: [B[i](A[j](args[k] for k in range(len_A-1)),A[j](args[len_A+k] for k in range(len_A-1))) for i in range(len_B) for j in range(len_A)]
# #

# tp1 = lambda a, b, c, d: h1(h1(a, b), h1(c, d))
# tp2 = lambda a, b, c, d: h1(h2(a, b), h2(c, d))
# tp3 = lambda a, b, c, d: h2(h1(a, b), h1(c, d))
# tp4 = lambda a, b, c, d: h2(h2(a, b), h2(c, d))
#
# TP = [tp1, tp2, tp3, tp4] #HxH
# A = [tp1, tp2, tp3, tp4] #HxH
#
# i1 = lambda x, y: x
# i2 = lambda x, y: y
# B = [i1, i2] #I
#
# inputs2 = [1, 2, 3, 5, 7, 11, 13, 17]



# C = TensProd(I, IH)

# print(C(1,2,3,5,7,11,13,17))