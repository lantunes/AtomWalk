try:
    import cPickle as pickle
except ImportError:
    import pickle

from numpy import asarray

if __name__ == '__main__':
    # get atoms supported by Atom2Vec and filter the ABC2D6 compounds to be used for training based on those atoms
    a2v = ["N", "O", "S", "Se", "I", "Cl", "F", "Br", "H", "Li", "Na", "K", "Rb", "Cs", "Tl", "Be", "Mg", "Sr", "Ba", "Ca",
    "Pb", "B", "Al", "Bi", "Ga", "Sb", "Sn", "Te", "In", "As", "C", "Si", "Ge", "P"]

    periodictable_symbols = asarray(
        [0, 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
         'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co',
         'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
         'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
         'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
         'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
         'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
         'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No',
         'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Uuq', 'Uup', 'Uuh', 'Uus', 'Uuo'])

    with open("../out/abc2d6-16/TrainingSet.pkl", 'rb') as pickle_file:
        inp = pickle.load(pickle_file, encoding="latin1")

    X = []
    y = []

    inp['T'] = inp['T'] / inp['N']
    for i in range(len(inp['T'])):

        atoms = [str(periodictable_symbols[e]) for e in inp['Z'][i]]

        skip = False
        for atom in atoms:
            if atom not in a2v:
                skip = True

        if not skip:
            X.append(" ".join(atoms))
            y.append(inp['T'][i])  # Formation energy (eV/atom)

        # print('---', i, '---')
        # print('Formation energy (eV/atom): ', inp['T'][i])
        # print('Coordinates: ')
        # for j in range(len(inp['Co'][i])):
        #     print(inp['Co'][i, j, 0], inp['Co'][i, j, 1], inp['Co'][i, j, 2])
        # print('Cell: ')
        # for j in range(len(inp['Ce'][i])):
        #     print(inp['Ce'][i, j, 0], inp['Ce'][i, j, 1], inp['Ce'][i, j, 2])
        # print('Atoms: ')
        # print(" ".join(str(periodictable_symbols[e]) for e in inp['Z'][i]))
        # print('Representation: ')
        # for j in range(len(inp['X'][i])):
        #     print(inp['X'][i, j, 0], inp['X'][i, j, 1])

    with open("../out/abc2d6_training.pkl", 'wb') as pickle_file:
        pickle.dump((X, y), pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
