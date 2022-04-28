import math
from scipy.optimize import minimize
import numpy as np

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
        e_dk = calculate_e_dk(dk, spectra, error_vector)
        e_dks.append(e_dk)
        prior = math.prod([priors[c] for c in dk])
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

    print(f'diagnoses and probabilities:')
    for i, _ in enumerate(lz_diagnoses):
        print(f'{lz_diagnoses[i]}: {lz_probabilities[i]}')

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

def calculate_e_dk(dk, spectra, error_vector):
    funcArr = ['(-1)']
    objective = None
    active_vars = [False] * len(spectra[0])

    # get the active vars in this diagnosis
    for i, e in enumerate(error_vector):
        for j, c in enumerate(spectra[i]):
            if spectra[i][j] == 1 and j in dk:
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

"""
def available(wanted_resource, occupied_resources):
    for ocr in occupied_resources:
        if wanted_resource == ocr:
            return False
    return True

#############################################################
# Methods for calculating diagnoses and their probabilities #
#############################################################
def calculate_dichotomy_matrix(spectra, error_vector):
    dichotomy_matrix = [[],  # n11
                        [],  # n10
                        [],  # n01
                        []]  # n00
    for cj in range(len(spectra[0])):
        n11, n10, n01, n00 = 0, 0, 0, 0
        cj_vector = [spectra[i][cj] for i in range(len(spectra))]
        for i in range(len(cj_vector)):
            if cj_vector[i] == 1 and error_vector[i] == 1:
                n11 += 1
            elif cj_vector[i] == 1 and error_vector[i] == 0:
                n10 += 1
            elif cj_vector[i] == 0 and error_vector[i] == 1:
                n01 += 1
            else:
                n00 += 1
        dichotomy_matrix[0].append(n11)
        dichotomy_matrix[1].append(n10)
        dichotomy_matrix[2].append(n01)
        dichotomy_matrix[3].append(n00)
    return dichotomy_matrix

def calculate_diagnoses_and_probabilities_barinel_amir(spectra, error_vector, kwargs, simulations):
    # Calculate prior probabilities
    priors = methods[kwargs['mfcp']](spectra,
                                                              error_vector,
                                                              kwargs,
                                                              simulations)

    # calculate optimized probabilities
    failed_tests = list(
        map(lambda test: list(enumerate(test[0])), filter(lambda test: test[1] == 1, zip(spectra, error_vector))))
    used_components = dict(enumerate(sorted(reduce(set.__or__, map(
        lambda test: set(map(lambda comp: comp[0], filter(lambda comp: comp[1] == 1, test))), failed_tests), set()))))
    optimizedMatrix = FullMatrix()
    optimizedMatrix.set_probabilities([x[1] for x in enumerate(priors) if x[0] in used_components])
    newErr = []
    newMatrix = []
    used_tests = []
    for i, (test, err) in enumerate(zip(spectra, error_vector)):
        newTest = list(map(lambda i: test[i], sorted(used_components.values())))
        if 1 in newTest:  ## optimization could remove all comps of a test
            newMatrix.append(newTest)
            newErr.append(err)
            used_tests.append(i)
    optimizedMatrix.set_matrix(newMatrix)
    optimizedMatrix.set_error(newErr)
    used_tests = sorted(used_tests)

    # rename back the components of the diagnoses
    Opt_diagnoses = optimizedMatrix.diagnose()
    diagnoses = []
    for diag in Opt_diagnoses:
        diag = diag.clone()
        diag_comps = [used_components[x] for x in diag.diagnosis]
        diag.diagnosis = list(diag_comps)
        diagnoses.append(diag)

    # transform diagnoses to 2 lists like the default barinel
    t_diagnoses, t_probabilities = [], []
    for d in diagnoses:
        t_diagnoses.append(d.diagnosis)
        t_probabilities.append(d.probability)

    # normalize probabilities and order them
    probabilities_sum = sum(t_probabilities)
    for i, probability in enumerate(t_probabilities):
        t_probabilities[i] = t_probabilities[i] / probabilities_sum
    z_probabilities, z_diagnoses = zip(*[(d, p) for d, p in sorted(zip(t_probabilities, t_diagnoses))])
    lz_diagnoses = list(z_diagnoses)
    lz_probabilities = list(z_probabilities)
    lz_diagnoses.reverse()
    lz_probabilities.reverse()

    print(f'oracle: {[a.num for a in simulations[0].agents if a.is_faulty]}')

    print(f'diagnoses and probabilities:')
    for i, _ in enumerate(lz_diagnoses):
        print(f'{lz_diagnoses[i]}: {lz_probabilities[i]}')

    return lz_diagnoses, lz_probabilities


#############################################################
#              Methods for calculating priors               #
#############################################################
def populate_intersections_table(num_agents, simulations):
    # initialize intersections table
    intersections_table = np.zeros((num_agents, num_agents), dtype=int)

    # populate intersections table across the different simulations
    for i, simulation in enumerate(simulations):
        current_plans = simulation.plans
        for a in range(num_agents):
            for t in range(len(current_plans[a]) - 1):
                for a2 in range(num_agents):
                    if a2 != a:
                        for t2 in range(t + 1, len(current_plans[a2])):
                            avi = current_plans[a][t]
                            bruno = current_plans[a2][t2]
                            if current_plans[a][t] == current_plans[a2][t2]:
                                intersections_table[a][a2] += 1
    return intersections_table


def calculate_priors_one(spectra, error_vector, kwargs, simulations):
    p = 1
    priors = [p for _ in range(len(spectra[0]))]  # priors
    return priors

def calculate_priors_static(spectra, error_vector, kwargs, simulations):
    p = 0.1
    priors = [p for _ in range(len(spectra[0]))]  # priors
    return priors
"""
