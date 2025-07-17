import numpy as np
from math import prod
from scipy.optimize import minimize

from APPETITE import Constants as constants

def get_total_likelihood(diagnosis: np.ndarray,
                         healthiness_probabilities: np.ndarray,
                         spectrum: np.ndarray,
                         fuzzy_error_vector: np.ndarray
 ) -> float:
    """
    Get the likelihood of the diagnosis.

    Parameters:
    diagnosis (ndarray): The diagnosis.
    healthiness_probabilities (ndarray): The healthiness probabilities.
    spectrum (ndarray): The spectrum.
    fuzzy_error_vector (ndarray): The fuzzy error vector.

    Returns:
    float: The likelihood of the diagnosis.
    """
    def get_single_test_likelihood(participated_components: np.ndarray,
                                   participation_vector: np.ndarray,
                                    fuzzy_error: float
        ) -> float:
        """"
        Get the likelihood of the single test.
        """
        transaction_goodness = healthiness_probabilities[participated_components].prod()
        transaction_goodness *= participation_vector[participated_components].prod()
        return fuzzy_error * (1 - transaction_goodness) + (1 - fuzzy_error) * transaction_goodness
    get_diagnosis_participated_components = lambda test_participation_vector: diagnosis[test_participation_vector[diagnosis] > 0]
    tests_diagnosis_components = map(get_diagnosis_participated_components, spectrum)
    tests_likelihoods = map(get_single_test_likelihood, tests_diagnosis_components, spectrum, fuzzy_error_vector)
    return -prod(tests_likelihoods) # Maximize the likelihood

def rank_diagnosis(diagnosis: np.ndarray,
                   spectrum: np.ndarray,
                   fuzzy_error_vector: np.ndarray,
                   components_prior_probabilities: np.ndarray
 ) -> float:
    """
    Rank the diagnosis.

    Parameters:
    diagnosis (ndarray): The diagnosis.
    spectrum (ndarray): The spectrum.
    fuzzy_error_vector (ndarray): The fuzzy error vector.

    Returns:
        float: The rank of the diagnosis.
    """
    components_count = spectrum.shape[1]
    components_prior_probabilities = components_prior_probabilities.copy()
    np.vectorized_flip_probability = np.vectorize(lambda spectrum_index, probability: probability if spectrum_index in diagnosis else 1 - probability)
    components_prior_probabilities = np.vectorized_flip_probability(np.arange(components_count), components_prior_probabilities)
    prior_probability = components_prior_probabilities.prod()
    healthiness_probabilities = np.full(components_count, 0.5)
    healthiness_bounds = [(0, 1) for _ in range(components_count)]
    likelihood_objective_function = lambda healthiness_probabilities: get_total_likelihood(diagnosis, healthiness_probabilities, spectrum, fuzzy_error_vector)
    mle_model = minimize(likelihood_objective_function, healthiness_probabilities, bounds=healthiness_bounds, options={"maxiter": 1000})
    # Get maximum likelihood estimation
    maximum_likelihood = -mle_model.fun
    return maximum_likelihood * prior_probability

def rank_diagnoses(spectrum: np.ndarray,
                   diagnoses: list[np.ndarray],
                   components_prior_probabilities: np.ndarray = None
 ) -> list[tuple[np.ndarray, float]]:
    """
    Rank the diagnoses.

    Parameters:
    spectrum (ndarray): The spectrum.
    diagnoses (list[ndarray]): The diagnoses.
    fuzzy_error_vector (ndarray): The fuzzy error vector.

    Returns:
    list[tuple[ndarray, float]]: The ranked diagnoses.
    """
    spectrum, fuzzy_error_vector = spectrum[:, :-1], spectrum[:, -1]
    if components_prior_probabilities is None:
        components_prior_probabilities = np.full(spectrum.shape[1], constants.BARINEL_COMPONENT_PRIOR_PROBABILITY)
    return [(diagnosis, rank_diagnosis(diagnosis, spectrum, fuzzy_error_vector, components_prior_probabilities)) for diagnosis in diagnoses]