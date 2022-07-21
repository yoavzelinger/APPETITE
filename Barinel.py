import math
from scipy.optimize import minimize
import numpy as np
from functools import reduce
import operator

from pysat.examples.hitman import Hitman

def calculate_diagnoses_and_probabilities_barinel_shaked(spectra,  # np array [number of tests, number of components] - binary
                                                      error_vector,  # np array [number of tests] - binary
                                                      priors):  # np array [number of components] - float
    # get all conflicts
    errors_mask = error_vector[:] == 1
    components = np.arange(spectra.shape[1]) +1
    comps_in_tests = spectra*components
    comps_in_failed_tests = comps_in_tests[errors_mask,:]
    conflicts = set()
    for comps in comps_in_failed_tests:
        conf = comps[comps > 0] -1
        conflicts.add(tuple(conf))
    # print(f"conflicts = {conflicts}")

    # get all diagnoses by minimal hitting set on conflicts
    diagnoses = []
    with Hitman(bootstrap_with=conflicts, htype='sorted') as hitman:
        for hs in hitman.enumerate():
            dk = list(map(int,hs))
            diagnoses.append(dk)

    # calculate probabilities
    probabilities = np.zeros(len(diagnoses))
    e_dks = []
    for i, dk in enumerate(diagnoses):
        e_dk = calculate_e_dk(dk, spectra, error_vector)
        # print(f"pr(e | d = {dk}) = {e_dk}")
        e_dks.append(e_dk)
        prior = np.prod(priors[dk])  # multiply all diagnosis prior probabilities
        probabilities[i] = prior * e_dk

    # normalize probabilities
    probabilities_sum = probabilities.sum()
    probabilities /= probabilities_sum

    # order diagnoses by probability
    d_order = np.argsort(-probabilities)  # highest prob first
    o_probabilities = probabilities[d_order]
    o_diagnoses = []
    for i in d_order:
        o_diagnoses.append(diagnoses[i])

    # return ordered and normalized diagnoses and probabilities
    return o_diagnoses, o_probabilities

def calculate_e_dk(dk, spectra, error_vector):
    n_tests = spectra.shape[0]
    n_components = spectra.shape[1]

    # get the active vars in this diagnosis
    active_vars = np.zeros(n_components)  # var for each component
    spectra_dk = spectra[:,dk]
    dk_active_mask = np.sum(spectra_dk, axis=0) > 0  # true if component form dk was in any test
    active_vars[dk] = np.ones(len(dk))*dk_active_mask
    n_active_vars = int(active_vars.sum())

    # re-labeling variables to conform to scipy's requirements
    renamed_vars = -np.ones(n_components)
    var_names = np.arange(n_active_vars)
    renamed_vars[active_vars > 0] = var_names

    # building the target function as a string
    func = "(-1)" # we want to maximize func, so we minimize -func
    for i in range(n_tests):  # iterate tests
        fa = "1*"
        for j in dk:  # iterate components in diagnosis
            if spectra[i][j] == 1:  # comp in test
                fa = fa + f"x[{int(renamed_vars[j])}]*"  # add healthy var to function
        fa = fa[:-1]  # remove last *
        if error_vector[i] == 1:
            fa = "*(1-" + fa + ")"
        else:
            fa = "*(" + fa + ")"
        func = func + fa

    # using dynamic programming to initialize the target function - x is vector
    objective = eval(f'lambda x: {func}')
    # print(func)

    # building bounds over the variables and the initial health vector
    #b = (0.0, 1.0)  # probabilities
    b = (0.00001, 0.99999)  # probabilities - not 0 or 1
    initial_h = 0.5
    bnds = [b]*n_active_vars
    h0 = np.ones(n_active_vars)*initial_h

    # solving the minimization problem
    sol = minimize(objective, h0, method="L-BFGS-B", bounds=bnds, tol=0.001, options={'maxiter': 100})

    # print(sol.x)
    # print(-sol.fun)
    return -sol.fun

def calculate_diagnoses_and_probabilities_barinel_avi(spectra,  # list(list) (number of tests, number of components)
                                                      error_vector,  # list() number of tests
                                                      priors):  # list in the length of components [float]
    # # Calculate diagnoses using hitting sets with CDS
    conflicts = []
    for i in range(len(error_vector)):
        if error_vector[i] == 1:
            c = []
            for j in range(len(spectra[i])):
                if spectra[i][j] == 1:
                    c.append(j)
            conflicts.append(c)
    diagnoses = conflict_directed_search(conflicts=conflicts)

    # # calculate probabilities
    probabilities = [0.0 for _ in diagnoses]
    e_dks = []
    for i, dk in enumerate(diagnoses):
        e_dk = calculate_e_dk_avi(dk, spectra, error_vector)
        e_dks.append(e_dk)
        # prior = math.prod([priors[c] for c in dk])
        prior = reduce(operator.mul, [priors[c] for c in dk], 1)
        probabilities[i] = prior * e_dk

    # normalize probabilities and order them
    probabilities_sum = sum(probabilities)
    for i, probability in enumerate(probabilities):
        probabilities[i] = probabilities[i] / probabilities_sum
    z_probabilities, z_diagnoses = zip(*[(d, p) for d, p in sorted(zip(probabilities, diagnoses))])
    lz_diagnoses = list(z_diagnoses)
    lz_probabilities = list(z_probabilities)
    lz_diagnoses.reverse()
    lz_probabilities.reverse()

    # print diagnoses and probabilities
    """
    print(f'diagnoses and probabilities:')
    for i, _ in enumerate(lz_diagnoses):
        print(f'{lz_diagnoses[i]}: {lz_probabilities[i]}')
    """

    # return ordered and normalized diagnoses and probabilities
    return lz_diagnoses, lz_probabilities

def conflict_directed_search(conflicts):
    diagnoses = []
    new_diagnoses = [[conflicts[0][i]] for i in range(len(conflicts[0]))]
    for conflict in conflicts[1:]:
        diagnoses = new_diagnoses
        new_diagnoses = []
        while len(diagnoses) != 0:
            diagnosis = diagnoses.pop(0)
            intsec = list(set(diagnosis) & set(conflict))
            if len(intsec) == 0:
                new_diags = [diagnosis + [c] for c in conflict]

                def filter_supersets(new_diag):
                    for d in diagnoses + new_diagnoses:
                        if set(d) <= set(new_diag):
                            return False
                    return True

                filtered_new_diags = list(filter(filter_supersets, new_diags))
                new_diagnoses += filtered_new_diags
            else:
                new_diagnoses.append(diagnosis)
    diagnoses = new_diagnoses
    return diagnoses

def calculate_e_dk_avi(dk, spectra, error_vector):
    funcArr = ['(-1)']
    objective = None
    active_vars = [False] * len(spectra[0])

    # get the active vars in this diagnosis
    for i, e in enumerate(error_vector):
        for j, c in enumerate(spectra[i]):
            if spectra[i][j] == 1 and j in dk: # j in diagnosis & participated in test
                active_vars[j] = True

    # re-labeling variables to conform to scipy's requirements
    index_rv = 0
    renamed_vars = {}
    for i, av in enumerate(active_vars):
        if av:
            renamed_vars[str(i)] = index_rv
            index_rv += 1

    # building the target function as a string
    for i, e in enumerate(error_vector):
        fa = "1*"
        for j, c in enumerate(spectra[i]):
            if spectra[i][j] == 1 and j in dk:
                fa = fa + f"x[{renamed_vars[str(j)]}]*"
        fa = fa[:-1]
        if error_vector[i] == 1:
            fa = "*(1-" + fa + ")"
        else:
            fa = "*(" + fa + ")"
        funcArr.append(fa)

    # using dynamic programming to initialize the target function
    func = ""
    for fa in funcArr:
        func = func + fa
    objective = eval(f'lambda x: {func}')

    # building bounds over the variables
    # and the initial health vector
    b = (0.0, 1.0)
    initial_h = 0.5
    bnds = []
    h0 = []
    for av in active_vars:
        if av:
            bnds.append(b)
            h0.append(initial_h)

    # solving the minimization problem
    h0 = np.array(h0)
    sol = minimize(objective, h0, method="L-BFGS-B", bounds=bnds, tol=1e-3, options={'maxiter': 100})

    return -sol.fun

if __name__ == '__main__':
    priors = np.ones(3)
    """
    spectra1 = np.array([[1, 0, 1],
                        [0, 1, 1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [1, 1, 0]])
    error_vector1 = np.array([1, 1, 1, 1, 0])

    d,p = calculate_diagnoses_and_probabilities_barinel_shaked(spectra1,error_vector1,priors)
    print(f"diagnoses = {d}")
    print(f"probabilities = {p}")
    """

    spectra2 = np.array([[1, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1]])
    error_vector2 = np.array([1, 1, 1, 0])

    d, p = calculate_diagnoses_and_probabilities_barinel_shaked(spectra2, error_vector2, priors)
    #d, p = calculate_diagnoses_and_probabilities_barinel_avi(spectra2.tolist(), error_vector2.tolist(), priors.tolist())
    print(f"diagnoses = {d}")
    assert len(d) == 2, "wrong number of diagnoses"
    assert sorted(d[0]) == [0, 1], "wrong diagnosis order"
    assert sorted(d[1]) == [0, 2], "wrong diagnosis order"
    print(f"probabilities = {p}")
    assert len(p) == 2
    # assert abs(p[0] - 0.839) < 0.01, f"prob should be 0.839, but it is {p[0]}"
    # assert abs(p[1] - 0.161) < 0.01, f"prob should be 0.161, but it is {p[1]}"

    # priors = np.ones(5)
    # priors = np.array([0.01, 0.05, 0.1, 0.1, 0.2])
    # spectra2 = np.array([[1, 1, 1, 0, 0],
    #                      [1, 1, 0, 1, 1],
    #                      [1, 1, 0, 1, 1]])
    #                      # [1, 1, 1, 0, 0],
    #                      # [1, 1, 1, 0, 0],
    #                      # [1, 1, 1, 0, 0]])
    # error_vector2 = np.array([1, 1, 1])#, 0, 0, 0])
    # d, p = calculate_diagnoses_and_probabilities_barinel_shaked(spectra2, error_vector2, priors)
    # print(f"diagnoses = {d}")
    # print(f"probabilities = {p}")
