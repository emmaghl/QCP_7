import numpy as np

#Logan's start!
inputs = [1, 2, 3, 5]
print(inputs)

# Hadamard
h1 = lambda x, y: (x + y) / np.sqrt(2)
h2 = lambda x, y: (x - y) / np.sqrt(2)

# Identity
i1 = lambda x, y: x
i2 = lambda x, y: y

# Tensor Product of I and H

tp1 = lambda a, b, c, d: i1(h1(a, b), h1(c, d))
tp2 = lambda a, b, c, d: i1(h2(a, b), h2(c, d))
tp3 = lambda a, b, c, d: i2(h1(a, b), h1(c, d))
tp4 = lambda a, b, c, d: i2(h2(a, b), h2(c, d))

TP = [tp1, tp2, tp3, tp4]

outputs = []
for i in range(0, len(TP)):
    outputs.append(TP[i](inputs[0], inputs[1], inputs[2], inputs[3]))
print(outputs)

# Tensor Product of H and I

tp1 = lambda a, b, c, d: h1(i1(a, b), i1(c, d))
tp2 = lambda a, b, c, d: h1(i2(a, b), i2(c, d))
tp3 = lambda a, b, c, d: h2(i1(a, b), i1(c, d))
tp4 = lambda a, b, c, d: h2(i2(a, b), i2(c, d))

#TP = lambda *args: [idn[i](h[j](args[0],args[1]),h[j](args[2],args[3])) for i in range(2) for j in range(2)]

TP = [tp1, tp2, tp3, tp4]

outputs = []
for i in range(0, len(TP)):
    outputs.append(TP[i](inputs[0], inputs[1], inputs[2], inputs[3]))
# print(outputs)

# Tensor Product of H and H

tp1 = lambda a, b, c, d: h1(h1(a, b), h1(c, d))
tp2 = lambda a, b, c, d: h1(h2(a, b), h2(c, d))
tp3 = lambda a, b, c, d: h2(h1(a, b), h1(c, d))
tp4 = lambda a, b, c, d: h2(h2(a, b), h2(c, d))

TP = [tp1, tp2, tp3, tp4]

outputs = []
for i in range(0, len(TP)):
    outputs.append(TP[i](inputs[0], inputs[1], inputs[2], inputs[3]))
# print(outputs)


# Tensor Product of TP(H,H) and I
inputs2 = [1, 2, 3, 5, 7, 11, 13, 17]
# print(inputs2)

tp1 = lambda a, b, c, d, e, f, g, h: TP[0](i1(a, b), i1(c, d), i1(e, f), i1(g, h))
tp2 = lambda a, b, c, d, e, f, g, h: TP[0](i2(a, b), i2(c, d), i2(e, f), i2(g, h))
tp3 = lambda a, b, c, d, e, f, g, h: TP[1](i1(a, b), i1(c, d), i1(e, f), i1(g, h))
tp4 = lambda a, b, c, d, e, f, g, h: TP[1](i2(a, b), i2(c, d), i2(e, f), i2(g, h))
tp5 = lambda a, b, c, d, e, f, g, h: TP[2](i1(a, b), i1(c, d), i1(e, f), i1(g, h))
tp6 = lambda a, b, c, d, e, f, g, h: TP[2](i2(a, b), i2(c, d), i2(e, f), i2(g, h))
tp7 = lambda a, b, c, d, e, f, g, h: TP[3](i1(a, b), i1(c, d), i1(e, f), i1(g, h))
tp8 = lambda a, b, c, d, e, f, g, h: TP[3](i2(a, b), i2(c, d), i2(e, f), i2(g, h))

TP2 = [tp1, tp2, tp3, tp4, tp5, tp6, tp7, tp8]

outputs = []
for i in range(0, len(TP2)):
    outputs.append(TP2[i](1, 2, 3, 5, 7, 11, 13, 17))


# print(outputs)


# Iterative version:

def H():
    h1 = lambda x, y: (x + y) / np.sqrt(2)
    h2 = lambda x, y: (x - y) / np.sqrt(2)
    return [h1, h2]


def I():
    i1 = lambda x, y: x
    i2 = lambda x, y: y
    return [i1, i2]

TP = lambda a,b,c,d: [idn[i](h[j](a,b),h[j](c,d)) for i in range(2) for j in range(2)]
print(TP)

def TensorProd():

'''
def TP(m1, m2):
    tp = []
    for i in range(0, len(m1)):
        for j in range(0, len(m2)):
            tp.append(lambda a, b, c, d: m1[i](m2[j](a, b), m2[j](c, d)))

    return tp


def TP2(m1, m2):
    tp = list(map((lambda x: x), m1))
    print(tp)
'''

#tenP = TP2(I(), H())



###### emma fiddling
# Hadamard
h1 = lambda x, y: (x + y) / np.sqrt(2)
h2 = lambda x, y: (x - y) / np.sqrt(2)

H = [h1, h2]

# Identity
i1 = lambda x, y: x
i2 = lambda x, y: y

I = [i1, i2]

# # Tensor Product of I and H
#
# tp1 = lambda a, b, c, d: i1(h1(a, b), h1(c, d))
# tp2 = lambda a, b, c, d: i1(h2(a, b), h2(c, d))
# tp3 = lambda a, b, c, d: i2(h1(a, b), h1(c, d))
# tp4 = lambda a, b, c, d: i2(h2(a, b), h2(c, d))
#
# TP = [tp1, tp2, tp3, tp4]
#
# outputs = []
# for i in range(0, len(TP)):
#     outputs.append(TP[i](inputs[0], inputs[1], inputs[2], inputs[3]))
# print(outputs)
#
# # Tensor Product of H and I
#
# tp1 = lambda a, b, c, d: h1(i1(a, b), i1(c, d))
# tp2 = lambda a, b, c, d: h1(i2(a, b), i2(c, d))
# tp3 = lambda a, b, c, d: h2(i1(a, b), i1(c, d))
# tp4 = lambda a, b, c, d: h2(i2(a, b), i2(c, d))
# TP = [tp1, tp2, tp3, tp4]
#
# #emma fiddling
#
# def H():
#     h1 = lambda x, y: (x + y) / np.sqrt(2)
#     h2 = lambda x, y: (x - y) / np.sqrt(2)
#     return [h1, h2]
#
# def I():
#     i1 = lambda x, y: x
#     i2 = lambda x, y: y
#     return [i1, i2]

def TensProd(A, B):
    # a =
    # b =
    # TP2 = lambda *args: [I[i](H[j](args[0],args[1]),H[j](args[2],args[3])) for i in range(len(I())) for j in range(len(H))]
    # TP2 = lambda *args: [I[i](H[j](args[0],args[1], ..., args[len(H)]),H[j](args[len(H)+0],args[len(H)+1],..., args[len(H) + len(H)])) for i in range(len(I())) for j in range(len(H))]
    return lambda *args: [B[i](A[j](args[k] for k in range(len(A)-1)),A[j](args[len(A)+k] for k in range(len(A)-1))) for i in range(len(B)) for j in range(len(A))]

tp1 = lambda a, b, c, d: h1(h1(a, b), h1(c, d))
tp2 = lambda a, b, c, d: h1(h2(a, b), h2(c, d))
tp3 = lambda a, b, c, d: h2(h1(a, b), h1(c, d))
tp4 = lambda a, b, c, d: h2(h2(a, b), h2(c, d))

TP = [tp1, tp2, tp3, tp4] #HxH
A = [tp1, tp2, tp3, tp4] #HxH

i1 = lambda x, y: x
i2 = lambda x, y: y
B = [i1, i2] #I

inputs2 = [1, 2, 3, 5, 7, 11, 13, 17]

C = TensProd(A, B)
print(C(inputs2))

tp1 = lambda a, b, c, d, e, f, g, h: TP[0](i1(a, b), i1(c, d), i1(e, f), i1(g, h))
tp2 = lambda a, b, c, d, e, f, g, h: TP[0](i2(a, b), i2(c, d), i2(e, f), i2(g, h))
tp3 = lambda a, b, c, d, e, f, g, h: TP[1](i1(a, b), i1(c, d), i1(e, f), i1(g, h))
tp4 = lambda a, b, c, d, e, f, g, h: TP[1](i2(a, b), i2(c, d), i2(e, f), i2(g, h))
tp5 = lambda a, b, c, d, e, f, g, h: TP[2](i1(a, b), i1(c, d), i1(e, f), i1(g, h))
tp6 = lambda a, b, c, d, e, f, g, h: TP[2](i2(a, b), i2(c, d), i2(e, f), i2(g, h))
tp7 = lambda a, b, c, d, e, f, g, h: TP[3](i1(a, b), i1(c, d), i1(e, f), i1(g, h))
tp8 = lambda a, b, c, d, e, f, g, h: TP[3](i2(a, b), i2(c, d), i2(e, f), i2(g, h))

TP2 = [tp1, tp2, tp3, tp4, tp5, tp6, tp7, tp8]


# stop emma fiddling