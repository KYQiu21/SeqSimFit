import numpy as np
from utils import *


# models converting fitness proxy to fitness
def proxy2fitness(energy, ref_energy, beta):
    """
    Fitness function used in Teufel and Wilke, 2017, J R Soc Interface

    convert fitness proxy (energy, likelihood, etc.) to fitness
    energy: statistical energy or rosetta energy of the given sequence, this is described as
    selection pressure in Norn and Andre, 2023, PLoS Comput Biol
    ref_energy: the energy threshold at which the protein has lost 50% of its activity
    beta: the inverse temperature determining how hard or soft the fitness threshold is
          - a large beta turns the fitness function into a hard cut-off, producing a fitness
            of 0 for energy > ref_energy, and a fitness of 1 for energy < ref_energy
    """

    fitness = 1 / (1 + np.exp((energy - ref_energy) * beta))

    return fitness


# function to calculate esm-if likelihood
def esmif_likelihood_calculator(coords, mutant_seq, model, alphabet, chain_id):
    """
    calculate the esm-if sum likelihood of a given sequence and a given structure

    coords: protein structure as array
    mutant_seq: mutant sequence to be evaluated as string
    model: pre-loaded esm-if model
    chain_id: chain id of the studied structure
    alphabet: pre-loaded esm-if alphabet
    """

    prob_tokens = run_model(coords, mutant_seq, model, alphabet, chain_target=chain_id)
    _, score_mutant = score_variants(mutant_seq, prob_tokens, alphabet)
    score_mutant = np.nansum(score_mutant)
    return -score_mutant


# function to calculate esm2 likelihood
def esm2_likelihood_calculator(mutant_seq, model, alphabet, batch_converter):
    """
    calculate the esm2 likelihood of a given sequence
    """

    # import torch

    data = [("seq", mutant_seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    tokens = batch_tokens.clone().cuda()

    log_likelihoods = []

    for i in range(1, tokens.size(1) - 1):
        masked_tokens = tokens.clone()
        masked_tokens[0, i] = alphabet.mask_idx

        with torch.no_grad():
            logits = model(masked_tokens)['logits']

        log_probs = torch.log_softmax(logits[0, i], dim=0)

        token_id = tokens[0, i].item()
        log_likelihood = log_probs[token_id].item()
        log_likelihoods.append(log_likelihood)

    esm2_likelihood = -np.sum(log_likelihoods) / len(mutant_seq)
    return esm2_likelihood


# function to calculate progen likelihood
def progen_likelihood_caculator(mutant_seq, model, tokenizer):
    """
    calculate the progen2 likelihood of a given sequence
    """

    seq = '1' + mutant_seq + '2'
    reverse = lambda s: s[::-1]

    ll_lr_mean = ll(tokens=seq, model=model, tokenizer=tokenizer, reduction='mean')
    ll_rl_mean = ll(tokens=reverse(seq), model=model, tokenizer=tokenizer, reduction='mean')

    ll_mean = -0.5 * (ll_lr_mean + ll_rl_mean)

    return ll_mean


# function to calculate proteinmpnn cross entropy
def proteinmpnn_cce_calculator(sequence, logits):
    """
    calculate the conditional cross entropy of a given sequence based on a given backbone
    """
    import torch
    one_hot = one_hot_encode_sequence(sequence)
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    onehot_tensor = torch.tensor(one_hot, dtype=torch.int64)
    labels = torch.argmax(onehot_tensor, dim=1)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits_tensor, labels)

    return loss.item()


# function to calculate rosetta energy
def rosetta_dg_calculator(modelPose, mutate_resi_idx, mutate_resi_original_aa, mutate_resi_aa, pack_radius, scorefxn,
                          path=None):

    mp = Pose()
    mp.assign(modelPose)
    mutate_residue(mp, mutate_resi_idx, mutate_resi_original_aa, pack_radius=pack_radius)
    energy_original = scorefxn(mp)

    mp = Pose()
    mp.assign(modelPose)
    mutate_residue(mp, mutate_resi_idx, mutate_resi_aa, pack_radius=pack_radius)
    energy_mutant = scorefxn(mp)
    # mp.dump_pdb(path)

    return energy_original, energy_mutant, mp


# function to calculate DCA energy
def plmdca_energy_calculator(sequence, J, h):
    """
    Compute the statistical energy of a sequence given J and h.

    Args:
        sequence (np.ndarray): Sequence as a 1D array of integers (1-based amino acid indices).
        J (np.ndarray): Coupling matrix of shape (L, L, q, q).
        h (np.ndarray): Field matrix of shape (L, q).

    Returns:
        float: The statistical energy of the sequence.
    """
    L = len(sequence)
    E = 0.0

    # Add field part
    for i, a in enumerate(sequence):
        E += h[i, a - 1]  # Adjust to 0-based index for amino acids

    # Add coupling part
    for i in range(L):
        for j in range(i + 1, L):
            a = sequence[i] - 1  # Adjust to 0-based index
            b = sequence[j] - 1
            E += J[i, j, a, b]

    return -E


def multi_plmdca_energy_calculator(alignment_matrix, J_matrix, H_matrix):
    """
    Compute energies for all sequences in the alignment matrix
    """
    return np.array([
        plmdca_energy_calculator(alignment_matrix[:, i], J_matrix, H_matrix)
        for i in range(alignment_matrix.shape[1])
    ])
