import numpy as np
from Bio import SeqIO
# import torch
import random
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.toolbox import *
from pyrosetta.rosetta.protocols.relax import FastRelax
pyrosetta.init("-out:level 0")
init("-mute core.pack.pack_rotamers core.pack.task")

# esm-if related
def run_model(coords, sequence, model, alphabet, cmplx=False, chain_target='A'):

    from esm.inverse_folding.util import load_structure, extract_coords_from_structure,CoordBatchConverter
    import torch

    device = next(model.parameters()).device

    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, sequence)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    # print("prev_output_tokens", prev_output_tokens.shape, prev_output_tokens)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)

    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)

    logits_swapped = torch.swapaxes(logits, 1, 2)
    token_probs = torch.softmax(logits_swapped, dim=-1)

    return token_probs


def score_variants(sequence,token_probs,alphabet):

    aa_list = []
    wt_scores = []
    skip_pos = 0

    alphabetAA_L_D = {'-':0,'_' :0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
    alphabetAA_D_L = {v: k for k, v in alphabetAA_L_D.items()}

    for i,n in enumerate(sequence):
        aa_list.append(n+str(i+1))
        score_pos=[]
        for j in range(1,21):
            score_pos.append(masked_absolute(alphabetAA_D_L[j],i, token_probs, alphabet))
            if n == alphabetAA_D_L[j]:
                WT_score_pos=score_pos[-1]

        wt_scores.append(WT_score_pos)

    return aa_list, wt_scores


def masked_absolute(mut, idx, token_probs, alphabet):

    mt_encoded = alphabet.get_idx(mut)

    score = token_probs[0,idx, mt_encoded]
    return score.item()


# rosetta
def build_model(path, scorefxn='beta_nov16_cart'):

    print(path)
    # clean the structure
    cleanATOM(path)
    modelPose = Pose()
    modelPose = pose_from_pdb(path.replace('.pdb', '.clean.pdb'))

    # select a score function
    scorefxn = pyrosetta.create_score_function(scorefxn)

    # relax the structure
    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.constrain_relax_to_start_coords(True)
    relax.coord_constrain_sidechains(True)
    relax.ramp_down_constraints(False)
    relax.cartesian(True)
    relax.min_type("lbfgs_armijo_nonmonotone")  # for non-Cartesian scorefunctions use'dfpmin_armijo_nonmonotone'
    relax.set_scorefxn(scorefxn)
    relax.apply(modelPose)
    energy_reference = scorefxn(modelPose)

    # save the relaxed structure
    modelPose.dump_pdb(path.replace('.pdb', '.relax.pdb'))

    return modelPose, energy_reference

def save_pose(path, modelPose):

    modelPose.dump_pdb(path.replace('.pdb', f'{random.random()}.pdb'))


# plmdca
amino_acid_mapping = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
    'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16,
    'T': 17, 'V': 18, 'W': 19, 'Y': 20, '-': 21, 'X': 21
}


def vec_to_seq(mapping):
    """
    integers to string
    """

    sequence = "ACDEFGHIKLMNPQRSTVWY-"
    seq = ""
    for i in mapping:
        seq += sequence[i - 1]  # Convert index to correct amino acid
    return seq


def seq_to_vec(sequence):
    """
    string to integers
    """

    return np.array([amino_acid_mapping[aa] for aa in sequence])


letter_to_num = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
    'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
    'V': 18, 'W': 19, 'Y': 20, '-': 21  # 21: gaps and unrecognized symbols
}


def letter2num(c):
    return letter_to_num.get(c.upper(), 21)


def read_fasta_alignment(filename, max_gap_fraction=1.0):
    """
    Reads a FASTA alignment file and converts it into a numerical matrix.

    Args:
        filename (str): Path to the FASTA file.
        max_gap_fraction (float): Maximum allowed fraction of gaps in a sequence.

    Returns:
        np.ndarray: Integer matrix (L x N), where each column is a sequence.
    """
    sequences = []
    valid_indices = None
    fseqlen = 0

    for record in SeqIO.parse(filename, "fasta"):
        seq = str(record.seq).replace('X', '-')
        ngaps = seq.count('-')

        if valid_indices is None:
            valid_indices = [i for i, c in enumerate(seq) if c != '.']
            fseqlen = len(valid_indices)

        assert len(seq) == len(valid_indices), "Inputs are not aligned"
        tst_fseqlen = sum(1 for i in valid_indices if seq[i].isupper() or seq[i] == '-')
        assert tst_fseqlen == fseqlen, "Inconsistent inputs"

        if ngaps / fseqlen <= max_gap_fraction:
            sequences.append(seq)

    assert len(sequences) > 0, "No sequences passed the filter (max_gap_fraction={})".format(max_gap_fraction)

    Z = np.zeros((fseqlen, len(sequences)), dtype=np.int8)
    for col_idx, seq in enumerate(sequences):
        for row_idx, i in enumerate(valid_indices):
            Z[row_idx, col_idx] = letter2num(seq[i])

    return Z


# progen
def create_model(ckpt, fp16=True):
    from progen.modeling_progen import ProGenForCausalLM
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    from tokenizers import Tokenizer
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())


def ce(tokens, model, tokenizer):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            target = torch.tensor(tokenizer.encode(tokens).ids).to('cuda')
            logits = model(target, labels=target).logits

            # shift
            logits = logits[:-1, ...]
            target = target[1:]

            return cross_entropy(logits=logits, target=target).item()


def cross_entropy(logits, target, reduction='mean'):
    return torch.nn.functional.cross_entropy(input=logits, target=target, weight=None, size_average=None, reduce=None, reduction=reduction)


def log_likelihood(logits, target, reduction='mean'):
    return -cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), reduction=reduction)


def ll(tokens, model, tokenizer, f=log_likelihood, reduction='mean'):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            target = torch.tensor(tokenizer.encode(tokens).ids).to('cuda')
            logits = model(target, labels=target).logits

            # shift
            logits = logits[:-1, ...]
            target = target[1:]

            # remove terminals
            bos_token, eos_token = 3, 4
            if target[-1] in [bos_token, eos_token]:
                logits = logits[:-1, ...]
                target = target[:-1]

            assert (target == bos_token).sum() == 0
            assert (target == eos_token).sum() == 0

            # remove unused logits
            first_token, last_token = 5, 29
            logits = logits[:, first_token:(last_token + 1)]
            target = target - first_token

            assert logits.shape[1] == (last_token - first_token + 1)

            return f(logits=logits, target=target, reduction=reduction).item()


# ProteinMPNN related
def one_hot_encode_sequence(seq):
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

    aa_to_index = {aa: idx for idx, aa in enumerate(alphabet)}

    one_hot = np.zeros((len(seq), len(alphabet)), dtype=int)
    for i, aa in enumerate(seq):
        if aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1
        # else: remains zero for unknown aa
    return one_hot



