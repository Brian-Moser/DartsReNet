from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent concat')

STEPS = 8
CONCAT = 8

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]

DARTS_Vanilla = Genotype(recurrent=[('sigmoid', 0), ('sigmoid', 1), ('relu', 2), ('sigmoid', 3), ('relu', 4), ('identity', 5), ('identity', 6), ('relu', 6)], concat=range(1, 9))
DARTS_Sigmoid = Genotype(recurrent=[('relu', 0), ('sigmoid', 1), ('identity', 2), ('identity', 3), ('identity', 4), ('sigmoid', 4), ('identity', 4), ('relu', 4)], concat=range(1, 9))
DARTS_Directional = Genotype(recurrent=[('relu', 0), ('sigmoid', 1), ('relu', 2), ('relu', 3), ('relu', 4), ('relu', 5), ('relu', 6), ('sigmoid', 7)], concat=range(1, 9))
