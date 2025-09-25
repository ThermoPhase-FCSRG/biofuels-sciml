"""
UNIFAC group assignments for each compound in the database.

Parameters were taken from:
Magnussen, T.; Rasmussen, P.; Fredenslund, A. (1981)
“UNIFAC parameter table for prediction of liquid–liquid equilibria,”
Ind. Eng. Chem. Process Des. Dev., 20.

This dictionary specifies which UNIFAC subgroups are present
in each compound, and how many of each subgroup occur.
"""

GROUPS = {
    "benzene": {9: 6},
    "butanol": {1: 1, 2: 3, 14: 1},
    "cyclohexane": {2: 6},
    "cyclooctane": {2: 8},
    "decane": {1: 2, 2: 8},
    "dodecane": {1: 2, 2: 10},
    "ethanol": {1: 1, 2: 1, 14: 1},
    "ethyl acetate": {1: 2, 2: 1, 77: 1},
    "ethyl octanoate": {1: 2, 2: 7, 77: 1},
    "ethyl decanoate": {1: 2, 2: 9, 77: 1},
    "ethyl laurate": {1: 2, 2: 11, 77: 1},
    "ethyl linoleate": {1: 2, 2: 13, 6: 2, 77: 1},
    "ethyl myristate": {1: 2, 2: 13, 77: 1},
    "ethyl oleate": {1: 2, 2: 15, 6: 1, 77: 1},
    "ethyl palmitate": {1: 2, 2: 15, 77: 1},
    "ethyl stearate": {1: 2, 2: 17, 77: 1},
    "ethylbenzene": {9: 5, 12: 1, 1: 1},
    "ethyl propionate": {1: 2, 2: 2, 77: 1},
    "heptane": {1: 2, 2: 5},
    "hexane": {1: 2, 2: 4},
    "isobutanol": {1: 2, 3: 1, 2: 1, 14: 1},
    "isopropanol": {1: 2, 3: 1, 14: 1},
    "methanol": {15: 1},
    "methyl acetate": {1: 2, 77: 1},
    "methyl butyrate": {1: 2, 2: 2, 77: 1},
    "methyl decanoate": {1: 2, 2: 8, 77: 1},
    "methyl laurate": {1: 2, 2: 10, 77: 1},
    "methyl linoleate": {1: 2, 2: 12, 6: 2, 77: 1},
    "methyl myristate": {1: 2, 2: 12, 77: 1},
    "methyl octanoate": {1: 2, 2: 6, 77: 1},
    "methyl oleate": {1: 2, 2: 14, 6: 1, 77: 1},
    "methyl palmitate": {1: 2, 2: 14, 77: 1},
    "methyl propionate": {1: 2, 2: 1, 77: 1},
    "methyl stearate": {1: 2, 2: 16, 77: 1},
    "nonane": {1: 2, 2: 7},
    "octane": {1: 2, 2: 6},
    "propanol": {1: 1, 2: 2, 14: 1},
    "propyl acetate": {1: 2, 2: 2, 77: 1},
    "toluene": {9: 5, 11: 1},
    "undecane": {1: 2, 2: 9},
}
