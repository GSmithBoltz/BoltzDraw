"""Boltzmann weighting and Metropolis-style acceptance utilities.

Ported faithfully from ``waterkit.utils``. These functions drive the
stochastic placement order, position/orientation selection, and
accept/reject filters used by the water sampler. All randomness goes
through ``numpy.random`` so the behaviour is identical to the original
WaterKit implementation.
"""

import numpy as np

from .constants import DEFAULT_TEMPERATURE, KB
from .typing import BoolMask, Indices, Scalars


def boltzmann_probabilities(energies: Scalars, temperature: float = DEFAULT_TEMPERATURE) -> Scalars:
    """Return Boltzmann occupation probabilities for the given state ``energies``.

    Each state ``i`` is weighted by ``exp(-E_i / (KB * T))`` and the weights are
    normalised so they sum to one. If every weight underflows to zero (energies
    far too high for the temperature), an all-zero array is returned instead.

    Args:
        energies: Energies of the candidate states (kcal/mol).
        temperature: Temperature in Kelvin.

    Returns:
        Array of probabilities with the same length as ``energies``. All zeros
        if the partition function underflowed to zero.
    """
    energies = np.array(energies)

    d = np.exp(-energies / (KB * temperature))
    d_sum = np.sum(d)

    if d_sum > 0:
        p = d / d_sum
    else:
        # It means that energies are too high
        return np.zeros(energies.shape[0])

    return p


def boltzmann_choices(
    energies: Scalars,
    temperature: float = DEFAULT_TEMPERATURE,
    size: int | None = None,
) -> Indices:
    """Sample state indices proportional to their Boltzmann probabilities.

    Draws ``size`` indices without replacement from ``range(len(energies))``
    using the Boltzmann probabilities as weights. When ``size`` exceeds the
    number of states with non-zero probability, it is clamped to that count so
    ``numpy.random.choice`` does not raise.

    Args:
        energies: Energies of the candidate states (kcal/mol).
        temperature: Temperature in Kelvin.
        size: Number of indices to draw. Defaults to a single index.

    Returns:
        Array of selected state indices. An empty array if all probabilities
        are zero.
    """
    p = boltzmann_probabilities(energies, temperature)

    if np.sum(p) == 0.0:
        return np.array([])

    if size is None:
        size = 1

    if size > 1:
        # If some prob. in p are zero, ValueError: size of nonzero p is lower than size
        non_zero = np.count_nonzero(p)
        size = non_zero if non_zero < size else size

    i = np.random.choice(len(energies), size, False, p)

    return i


def boltzmann_acceptance_rejection(
    new_energies: Scalars,
    energy_cutoff: float | Scalars = 0.0,
    temperature: float = DEFAULT_TEMPERATURE,
) -> BoolMask:
    """Metropolis accept/reject mask for candidate ``new_energies``.

    Candidates whose energy is below the reference (``energy_cutoff``) are
    accepted outright. Unfavourable candidates are accepted with probability
    ``min(1, exp(-dE / (KB * T)))`` where ``dE`` is the energy increase over the
    reference, sampled against uniform random draws.

    Args:
        new_energies: Candidate energies (kcal/mol). Flattened before use.
        energy_cutoff: Reference energy/energies the candidates are compared
            against. A scalar reference is broadcast to all candidates.
        temperature: Temperature in Kelvin.

    Returns:
        Boolean array, one decision per candidate, ``True`` where accepted.
    """
    new_energies = np.ravel(new_energies)
    old_energies = np.ravel(energy_cutoff)

    decisions = new_energies < old_energies

    if decisions.all():
        return decisions
    else:
        unfavorable_indices = np.where(~decisions)

        if old_energies.size == 1:
            unfavorable_old_energies = old_energies
        else:
            unfavorable_old_energies = old_energies[unfavorable_indices]

        delta_e = new_energies[unfavorable_indices] - unfavorable_old_energies
        p_acc = np.minimum(1.0, np.exp(-delta_e / (KB * temperature)))
        r = np.random.rand(p_acc.shape[0])

        decisions[unfavorable_indices] = r <= p_acc

        return decisions
