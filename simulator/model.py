import numpy as np
import random


def kimura_fixation_prob(current_fitness, old_fitness, N):
    """
    The Kimura model for selection to compute the fixation probability of a proposed mutation
    current_fitness: fitness of the proposed mutant
    old_fitness: fitness of the current mutant
    N: population size
    """

    sel_coff = (current_fitness / old_fitness) - 1
    pacc = (1 - np.exp(-2 * sel_coff)) / (1 - np.exp(-4 * N * sel_coff))

    return pacc


# Function for MCMC fixation probability
def mcmc_fixation_prob(current_fitness, old_fitness, beta):
    delta = np.exp(-beta * (current_fitness - old_fitness))
    pacc = min(1, delta)
    return pacc


# Function to compute birth-death fixation probability
def birthdeath_fixation_prob(current_fitness, old_fitness, N):
    sel_coff = current_fitness / old_fitness
    pacc = (1 - (sel_coff ** 2)) / (1 - (sel_coff ** (2 * N)))
    return pacc


def accelerated_birthdeath_fixation_prob(current_fitness, old_fitness, N):
    """
    the accelerated algorithm to simulate under the birth-death model
    current_fitness: fitness of the proposed mutant
    old_fitness: fitness of the current mutant
    N: population size
    """

    if current_fitness > old_fitness:
        pacc = 1
    else:
        pacc = (current_fitness / old_fitness) ** (2 * N - 2)

    return pacc


def birthdeath_fixation_prob(current_fitness, old_fitness, N):
    """
    the algorithm to simulate under the birth-death model
    current_fitness: fitness of the proposed mutant
    old_fitness: fitness of the current mutant
    N: population size
    """

    pacc = (1 - (old_fitness / current_fitness) ** 2) / (1 - (old_fitness / current_fitness) ** (2*N))

    return pacc


def mutator(sequence, allowed_mutation_dict):
    """
    receive a sequence and propose a mutation according to a dictionary of allowed mutations
    """

    # gap index is not considered in the mutation process
    non_gap_indices = [i for i, aa in enumerate(sequence) if aa != '-']
    mutation_pos = random.choice(non_gap_indices)
    current_amino_acid = sequence[mutation_pos]

    possible_mutations = allowed_mutation_dict[mutation_pos]

    # avoid mutating to current aa but only when there are more than one aa
    if len(possible_mutations) > 1:
        possible_mutations = [aa for aa in possible_mutations if aa != current_amino_acid]

    # mutate to current when no candidate aa is in the list (gap)
    if len(possible_mutations) == 0:
        possible_mutations = [current_amino_acid]

    new_amino_acid = random.choice(possible_mutations)
    mutated_sequence = sequence[:mutation_pos] + new_amino_acid + sequence[mutation_pos + 1:]

    return mutated_sequence, mutation_pos, current_amino_acid, new_amino_acid
