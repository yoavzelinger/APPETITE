from numpy import int_, float_, full, arange, vectorize
from numpy.typing import NDArray

from scipy.optimize import minimize

from APPETITE.Constants import BARINEL_COMPONENT_PRIOR_PROBABILITY

def get_total_likelihood(diagnosis: NDArray[int_],
                         healthiness_probabilities: NDArray[float_],
                         spectra: NDArray[int_],
                         fuzzy_error_vector: NDArray[float_]
 ) -> float:
    """
    Get the likelihood of the diagnosis.

    Parameters:
    diagnosis (ndarray): The diagnosis.
    healthiness_probabilities (ndarray): The healthiness probabilities.
    spectra (ndarray): The spectra.
    fuzzy_error_vector (ndarray): The fuzzy error vector.

    Returns:
    float: The likelihood of the diagnosis.
    """
    def get_single_test_likelihood(participated_components: NDArray[int_],
                                    fuzzy_error: float
        ) -> float:
        """"
        Get the likelihood of the single test.
        """
        if participated_components.size == 0:
            return 0
        transaction_goodness = healthiness_probabilities[participated_components].prod()
        return fuzzy_error * (1 - transaction_goodness) + (1 - fuzzy_error) * transaction_goodness
    get_participated_components = lambda participation_vector: diagnosis[participation_vector[diagnosis] == 1]
    spectra_diagnosis_components = [get_participated_components(spectra_component) for spectra_component in spectra]
    tests_likelihoods = map(get_single_test_likelihood, spectra_diagnosis_components, fuzzy_error_vector)
    return -sum(tests_likelihoods) # Maximize the likelihood

def rank_diagnosis(diagnosis: NDArray[int_],
                   spectra: NDArray[int_],
                   fuzzy_error_vector: NDArray[float_],
                   components_prior_probabilities: NDArray[float_]
 ) -> float:
    """
    Rank the diagnosis.

    Parameters:
    diagnosis (ndarray): The diagnosis.
    spectra (ndarray): The spectra.
    fuzzy_error_vector (ndarray): The fuzzy error vector.

    Returns:
        float: The rank of the diagnosis.
    """
    components_count = spectra.shape[1]
    components_prior_probabilities = components_prior_probabilities.copy()
    flip_probability = lambda spectra_index, probability: probability if spectra_index in diagnosis else 1 - probability
    vectorized_flip_probability = vectorize(flip_probability)
    components_prior_probabilities = vectorized_flip_probability(arange(components_count), components_prior_probabilities)
    prior_probability = components_prior_probabilities.prod()
    healthiness_probabilities = [0.5 for _ in range(components_count)]
    healthiness_bounds = [(0, 1) for _ in range(components_count)]
    likelihood_objective_function = lambda healthiness_probabilities: get_total_likelihood(diagnosis, healthiness_probabilities, spectra, fuzzy_error_vector)
    mle_model = minimize(likelihood_objective_function, healthiness_probabilities, bounds=healthiness_bounds)
    # Get maximum likelihood estimation
    maximum_likelihood = -mle_model.fun
    return maximum_likelihood * prior_probability

def rank_diagnoses(spectra: NDArray[int_],
                   diagnoses: list[NDArray[int_]],
                   fuzzy_error_vector: NDArray[float_],
                   components_prior_probabilities: NDArray[float_] = None
 ) -> list[tuple[NDArray[int_], float]]:
    """
    Rank the diagnoses.

    Parameters:
    spectra (ndarray): The spectra.
    diagnoses (list[ndarray]): The diagnoses.
    fuzzy_error_vector (ndarray): The fuzzy error vector.

    Returns:
    list[tuple[ndarray, float]]: The ranked diagnoses.
    """
    if components_prior_probabilities is None:
        components_prior_probabilities = full(spectra.shape[1], BARINEL_COMPONENT_PRIOR_PROBABILITY)
    return [(diagnosis, rank_diagnosis(diagnosis, spectra, fuzzy_error_vector, components_prior_probabilities)) for diagnosis in diagnoses]