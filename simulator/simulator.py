from fitness import *
from utils import *
from model import *
from Bio import AlignIO
from collections import defaultdict
import os
import copy


class seq_simulator:
    """
    main class to perform sequence simulation

    Args:
    J_matrix: coupling matrix of the trained plmdca model
    H_matrix: field matrix of the trained plmdca model
    Ne: population size
    use_dca: whether to use dca in the simulation
    use_structure: whether to use structural stability in the simulation
    beta_dca: inverse of temperature of the dca fitness function
    beta_structure: inverse of temperature of the stability fitness function
    weight_dca: the weight of the dca energy in the fitness function
    weight_structure: the weight of the rosetta energy in the fitness function
    structure_energy_threshold: the highest structure energy allowed during the simulation
    model: which evolutionary model to use
    pdb_path: path to the full-length designed pdb file
    msa_path: path to the trimmed msa file using the natural sequence as query
    pairwise_path: path to the foldseek pairwise designed-natural alignment
    pack_radius: pack radius in rosetta every time a mutation is introduced
    scorefxn_name: which score function to use in calculate energy
    stop_mode: how the simulation completed
    mutation_space: control the space of introduced mutations, including 'unlimited', 'broad' or 'binary'
    """

    def __init__(self,
                 start_seq=None,
                 Ne=100,
                 evaluation_mode='rosetta',
                 beta=1.0,
                 ref_ratio=0.5,
                 model='birthdeath',
                 pdb_path=None,
                 msa_path=None,
                 J_matrix_path=None,
                 H_matrix_path=None,
                 esmif_model_path=None,
                 progen_model_path=None,
                 progen_tokenizer_path='/raven/u/kqiu/research_data/interpolation_framework/progen2/tokenizer.json',
                 mpnn_model_path=None,
                 chain_id=None,
                 pack_radius=10.0,
                 scorefxn_name='beta_nov16_cart',
                 stop_mode='step',
                 mutation_space='unlimited',
                 device='cuda',
                 step=10000,
                 step_limit=100000,
                 verbose=True):

        self.start_seq = start_seq
        self.Ne = Ne
        self.evaluation_mode = evaluation_mode
        self.beta = beta
        self.ref_ratio = ref_ratio
        self.model = model
        self.pdb_path = pdb_path
        self.msa_path = msa_path
        self.J_matrix_path = J_matrix_path
        self.H_matrix_path = H_matrix_path
        self.esmif_model_path = esmif_model_path
        self.progen_model_path = progen_model_path
        self.progen_tokenizer_path = progen_tokenizer_path
        self.mpnn_model_path = mpnn_model_path,
        self.chain_id = chain_id
        self.pack_radius = pack_radius
        self.scorefxn_name = scorefxn_name
        self.stop_mode = stop_mode
        self.mutation_space = mutation_space
        self.device = device
        self.step = step
        self.step_limit = step_limit
        self.verbose = verbose

        # preparation: load the model and calculate the reference fitness threshold according to the evaluation mode
        if evaluation_mode == 'rosetta':
            print("Rosetta energy is used to evolve sequences!")
            self._prepare_rosetta()

        if evaluation_mode == 'dca':
            print("DCA energy is used to evolve sequences!")
            self._prepare_dca()

        if evaluation_mode == 'esmif':
            print("ESM-IF likelihood is used to evolve sequences!")
            self._prepare_esmif()

        if evaluation_mode == 'esm2':
            print("ESM2 likelihood is used to evolve sequences!")
            self._prepare_esm2()

        if evaluation_mode == 'progen':
            print("ProGen likelihood is used to evolve sequences!")
            self._prepare_progen()

        if evaluation_mode == 'proteinmpnn':
            print("ProteinMPNN CCE is used to evolve sequences!")
            self._prepare_mpnn()

        self._prepare_seed()
        self._get_possible_mutations()

        print("All preparations done!")

    def _prepare_rosetta(self):
        """
        relax the input pdb structure, record the reference energy, prepare the score function and the pose
        """

        relax_pose, structure_energy_reference = build_model(self.pdb_path)
        # copy the initial relaxed pose for storage
        self.relax_pose = relax_pose
        self.updated_pose = relax_pose
        self.energy_reference = structure_energy_reference * self.ref_ratio
        self.scorefxn = pyrosetta.create_score_function(self.scorefxn_name)
        self.updated_energy = structure_energy_reference
        print("The rosetta energy threshold is ", self.energy_reference)
        print(f"The score function {self.scorefxn_name} is used to calculate energy")

    def _prepare_dca(self):
        """
        compute the average dca energy of the natural MSA
        """

        # read fasta
        msa_matrix = read_fasta_alignment(self.msa_path)
        energy_reference = multi_plmdca_energy_calculator(msa_matrix, self.J_matrix_path, self.H_matrix_path)
        energy_reference = np.mean(energy_reference)
        self.energy_reference = energy_reference * self.ref_ratio
        self.updated_energy = energy_reference
        print("The DCA energy threshold is ", self.energy_reference)

    def _prepare_esmif(self):
        """
        load the model and compute the wt esmif likelihood score
        """

        import esm
        from esm.inverse_folding.util import load_structure, extract_coords_from_structure, CoordBatchConverter
        # load model
        self.esmif_model, self.esmif_alphabet = esm.pretrained.load_model_and_alphabet(self.esmif_model_path)
        self.esmif_model.eval().cuda().requires_grad_(False)
        structure = load_structure(self.pdb_path, self.chain_id)
        self.coords, wt_seq = extract_coords_from_structure(structure)
        prob_tokens = run_model(self.coords, wt_seq, self.esmif_model, self.esmif_alphabet, chain_target=self.chain_id)
        aa_list, esmif_wt_likelihood = score_variants(wt_seq, prob_tokens, self.esmif_alphabet)
        esmif_wt_likelihood = -np.nansum(esmif_wt_likelihood)
        self.energy_reference = esmif_wt_likelihood * self.ref_ratio
        self.updated_energy = esmif_wt_likelihood
        print("The ESM-IF likelihood threshold is ", self.energy_reference)

    def _prepare_esm2(self):
        """
        load the model and compute the wt esm2 likelihood score
        """

        # load model
        self.esm2_model, self.esm2_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm2_model.eval().cuda()
        self.esm2_batch_converter = self.esm2_alphabet.get_batch_converter()
        esm2_likelihood = esm2_likelihood_calculator(self.start_seq)
        self.energy_reference = esm2_likelihood * self.ref_ratio
        self.updated_energy = esm2_likelihood
        print("The ESM2 likelihood threshold is ", self.energy_reference)

    def _prepare_progen(self):
        """
        load the model and compute the wt progen likelihood score
        """

        new_path = "/raven/u/kqiu/research_data/interpolation_framework/progen2"
        os.chdir(new_path)
        from tokenizers import Tokenizer
        from progen.modeling_progen import ProGenForCausalLM
        self.progen_model = create_model(ckpt=self.progen_model_path, fp16=True).to(self.device)
        self.progen_tokenizer = create_tokenizer_custom(file=self.progen_tokenizer_path)
        progen_likelihood = progen_likelihood_caculator(self.start_seq, self.progen_model, self.progen_tokenizer)
        self.energy_reference = progen_likelihood * self.ref_ratio
        self.updated_energy = progen_likelihood
        print("The ProGen likelihood threshold is ", self.energy_reference)

    def _prepare_mpnn(self):
        """
        load the model and compute the conditional cross entropy
        """

        new_path = "/raven/u/kqiu/tool/ProteinMPNN-main/"
        os.chdir(new_path)
        from protein_mpnn_utils import ProteinMPNN, StructureDatasetPDB, tied_featurize, parse_PDB
        import torch

        # first load the model
        checkpoint = torch.load(self.mpnn_model_path, map_location=self.device)
        self.mpnn_model = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128,
                                      num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.0, num_letters=21,
                                      k_neighbors=checkpoint['num_edges'])
        self.mpnn_model.to(self.device)
        self.mpnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.mpnn_model.eval()

        # then we compute a logit distribution of the given backbone
        pdb_dict_list = parse_PDB(self.pdb_path)
        dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None)
        all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9] == 'seq_chain']  # ['A','B', 'C',...]
        designed_chain_list = all_chain_list
        fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
        chain_id_dict = {pdb_dict_list[0]['name']: (designed_chain_list, fixed_chain_list)}
        fixed_positions_dict = None
        omit_AA_dict = None
        tied_positions_dict = None
        pssm_dict = None
        bias_by_res_dict = None
        BATCH_COPIES = 1
        NUM_BATCHES = 1

        for ix, protein in enumerate(dataset_valid):
            batch_clones = [copy.deepcopy(protein) for i in range(NUM_BATCHES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                batch_clones, self.device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict,
                bias_by_res_dict)

        log_conditional_probs_list = []
        log_unconditional_probs_list = []
        for j in range(BATCH_COPIES):
            torch.manual_seed(42)
            randn_1 = torch.zeros(chain_M.shape, device=X.device)  # 不加噪声
            # randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_conditional_probs = self.mpnn_model.conditional_probs(X, S, mask, chain_M * chain_M_pos, residue_idx,
                                                            chain_encoding_all, randn_1)
            log_unconditional_probs = self.mpnn_model.unconditional_probs(X, mask, residue_idx, chain_encoding_all)
            log_conditional_probs_list.append(log_conditional_probs.detach().cpu().numpy())
            log_unconditional_probs_list.append(log_unconditional_probs.detach().cpu().numpy())
        self.concat_log_p = np.concatenate(log_conditional_probs_list, 0)[0]  # [L, 21]
        self.concat_log_unp = np.concatenate(log_unconditional_probs_list, 0)[0] # [L, 21]

        mpnn_cce = proteinmpnn_cce_calculator(self.start_seq, self.concat_log_p)
        self.energy_reference = mpnn_cce * self.ref_ratio
        self.updated_energy = mpnn_cce
        print("The ProteinMPNN CCE threshold is ", self.energy_reference)

    def _prepare_seed(self, seq=None):
        """
        calculate and store information of the seed sequence
        """

        if seq is not None:
            self.updated_seq = seq
        else:
            self.updated_seq = self.start_seq
        if self.evaluation_mode == 'rosetta':
            self.updated_pose = self.relax_pose
        self.updated_fitness, self.updated_energy, _ = self.estimate_fitness(mutant_seq=self.updated_seq, initial=True)
        self.energy_reference = self.updated_energy * self.ref_ratio

        print("The initial sequence is ", self.updated_seq)
        print("The initial fitness is ", self.updated_fitness)

        # record the simulation
        self.all_step = 0
        self.complete_step = 0
        self.accepted_seq_list = [self.updated_seq]
        self.accpeted_seq_fitness_list = [self.updated_fitness]
        self.accpeted_seq_energy_list = [self.updated_energy]

    def _get_possible_mutations(self):

        """
        create a dictionary: key - residue index; value - a list of possible mutations
        three modes are available: msa - aa occurring in msa file; unlimited - all kinds of aa;
        binary - aa occurring in the pairwise alignment
        """

        self.allowed_mutation_dict = defaultdict(list)
        seq_len = len(self.start_seq)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        amino_acids_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',
                            'W', 'Y']

        if self.mutation_space == 'msa':
            msa = AlignIO.read(self.msa_path, "fasta")
            for pos in range(seq_len):
                for record in msa:
                    aa = record.seq[pos]
                    if aa != '-':
                        if aa not in self.allowed_mutation_dict[pos] and aa in amino_acids_list:
                            self.allowed_mutation_dict[pos].append(aa)

        if self.mutation_space == 'unlimited':
            for pos in range(seq_len):
                for aa in amino_acids:
                    self.allowed_mutation_dict[pos].append(aa)

    def estimate_fitness(self,
                         mutant_seq,
                         mutation_pos=None,
                         mutation_aa=None,
                         original_aa=None,
                         initial=False):

        """
        calculate the fitness of a proposed mutant sequence
        mutant_seq_string: mutant sequence in the amino acid string form
        mutant_seq_int: mutant sequence in the integer array form
        mutation_pos: the residue index to be mutated
        mutation_aa: the amino acid that the selected residue is mutated to
        original_aa: the amino acid that the selected residue is mutated from
        """

        fitness, proxy, updated_pose = None, None, None

        if self.evaluation_mode == 'rosetta' and not initial:
            original_structure_energy, structure_energy, updated_pose = rosetta_dg_calculator(self.updated_pose,
                                                                                              mutation_pos,
                                                                                              original_aa,
                                                                                              mutation_aa,
                                                                                              self.pack_radius,
                                                                                              self.scorefxn)
            fitness = proxy2fitness(structure_energy, self.energy_reference, self.beta)
            proxy = structure_energy

        if self.evaluation_mode == 'dca' and not initial:
            mutant_seq_int = seq_to_vec(mutant_seq)
            dca_energy = plmdca_energy_calculator(mutant_seq_int, self.J_matrix_path, self.H_matrix_path)
            fitness = proxy2fitness(dca_energy, self.energy_reference, self.beta)
            proxy = dca_energy

        if self.evaluation_mode == 'esmif' and not initial:
            esmif_likelihood = esmif_likelihood_calculator(self.coords, mutant_seq, self.esmif_model, self.esmif_alphabet, mself.chain_id, self.esmif_alphabet)
            fitness = proxy2fitness(esmif_likelihood, self.energy_reference, self.beta)
            proxy = esmif_likelihood

        if self.evaluation_mode == 'esm2' and not initial:
            esm2_likelihood = esm2_likelihood_calculator(mutant_seq, self.esm2_model, self.esm2_alphabet, self.esm2_batch_converter)
            fitness = proxy2fitness(esm2_likelihood, self.energy_reference, self.beta)
            proxy = esm2_likelihood

        if self.evaluation_mode == 'progen' and not initial:
            progen_likelihood = progen_likelihood_caculator(mutant_seq, self.progen_model, self.progen_tokenizer)
            fitness = proxy2fitness(progen_likelihood, self.energy_reference, self.beta)
            proxy = progen_likelihood

        if self.evaluation_mode == 'proteinmpnn' and not initial:
            mpnn_cce = proteinmpnn_cce_calculator(mutant_seq, self.concat_log_p)
            fitness = proxy2fitness(mpnn_cce, self.energy_reference, self.beta)
            proxy = mpnn_cce

        if self.evaluation_mode == 'proteinmpnn' and initial:
            mpnn_cce = proteinmpnn_cce_calculator(mutant_seq, self.concat_log_p)
            fitness = proxy2fitness(self.energy_reference / self.ref_ratio, self.energy_reference, self.beta)
            proxy = mpnn_cce

        if self.evaluation_mode == 'progen' and initial:
            progen_likelihood = progen_likelihood_caculator(mutant_seq, self.progen_model, self.progen_tokenizer)
            fitness = proxy2fitness(self.energy_reference/self.ref_ratio, self.energy_reference, self.beta)
            proxy = progen_likelihood

        if self.evaluation_mode == 'esm2' and initial:
            esm2_likelihood = esm2_likelihood_calculator(mutant_seq, self.esm2_model, self.esm2_alphabet, self.esm2_batch_converter)
            fitness = proxy2fitness(self.energy_reference / self.ref_ratio, self.energy_reference, self.beta)
            proxy = esm2_likelihood

        if self.evaluation_mode == 'rosetta' and initial:
            fitness = proxy2fitness(self.energy_reference / self.ref_ratio, self.energy_reference, self.beta)
            proxy = self.energy_reference / self.ref_ratio

        return fitness, proxy, updated_pose

    def fixation(self, old_fitness, current_fitness):

        """
        receive the old & current fitness and calculate the fixation probability
        """

        if self.model == 'birthdeath':
            pacc = accelerated_birthdeath_fixation_prob(current_fitness, old_fitness, self.Ne)

        if self.model == 'kimura':
            # importance sampling is applied to accelerate the simulation
            pacc = kimura_fixation_prob(current_fitness, old_fitness, self.Ne)

        return pacc

    def mutate(self, sequence):

        """
        Mutate an input sequence
        """

        mutant_sequence, mutation_pos, original_aa, mutation_aa = mutator(sequence, self.allowed_mutation_dict)
        # map the selected index to the full length sequence index
        rosetta_mutation_pos = mutation_pos+1
        print(f"Mutating {mutation_pos} from {original_aa} to {mutation_aa}")

        return mutant_sequence, rosetta_mutation_pos, mutation_pos, original_aa, mutation_aa

    def restart(self, seq=None):
        """
        clean all results in the previous simulation but remain other info to avoid reloading the object
        """

        if seq is not None:
            self._prepare_seed(seq)
        else:
            self._prepare_seed()
        print(len(self.accepted_seq_list), len(self.accpeted_seq_fitness_list), self.all_step, self.complete_step)
        assert len(self.accepted_seq_list) == 1 and len(self.accpeted_seq_fitness_list) == 1 and self.all_step == 0 and self.complete_step == 0
        # if self.evaluation_mode:
        #     print("Generating the initial pose for check!")
        #     save_pose(self.pdb_path, self.updated_pose)
        print("Reload the simulator and clean the previous simulation!")

    def simulate(self, step=1000):

        while self.complete_step < step and self.all_step < self.step_limit:

            # replace all old records using the updated records at the beginning of each iteration
            self.old_fitness = self.updated_fitness
            self.old_energy = self.updated_energy
            self.old_seq = self.updated_seq

            # propose a mutation
            mutant_sequence, rosetta_mutation_pos, mutation_pos, original_aa, mutation_aa = self.mutate(self.old_seq)
            print(rosetta_mutation_pos, mutation_pos, original_aa, mutation_aa)

            # evaluate fitness of the mutant sequence
            updated_fitness, updated_energy, updated_pose = self.estimate_fitness(mutant_sequence,
                                                                                  mutation_pos=rosetta_mutation_pos,
                                                                                  mutation_aa=mutation_aa,
                                                                                  original_aa=original_aa,
                                                                                  initial=False)

            # determine if the proposed mutation is accepted
            pacc = self.fixation(self.old_fitness, updated_fitness)
            if self.verbose:
                print("step ", self.complete_step+1, " old_fitness: ", self.old_fitness, " new_fitness: ", updated_fitness,
                      " old_energy: ", self.old_energy, " new_energy: ", updated_energy,
                      " pacc ", pacc)
            probabilities = [pacc, 1 - pacc]
            sample = random.choices([0, 1], probabilities)

            # accept the sequence under the energy threshold
            if sample[0] == 0:
                # if sample[0] == 0:
                if self.verbose:
                    print("Accept this mutation!")
                self.updated_seq = mutant_sequence
                self.updated_fitness = updated_fitness
                self.updated_energy = updated_energy
                self.accepted_seq_list.append(self.updated_seq)
                self.accpeted_seq_fitness_list.append(self.updated_fitness)
                self.accpeted_seq_energy_list.append(self.updated_energy)
                self.complete_step += 1

                if self.evaluation_mode == 'rosetta':
                    # update the pose!
                    self.updated_pose = updated_pose

            # reject the mutation
            else:
                if self.verbose:
                    print("Reject this mutation!")
                pass

            self.all_step += 1

        # if self.evaluation_mode:
        #     print("Generating the final pose for check!")
        #     save_pose(self.pdb_path, self.updated_pose)

    def simulate_on_a_tree(self, node, root_sequence):

        """
        simulate sequence evolution on a given phylogenetic tree - we just need to treat this task as separate
        independent 'simulate' processes and complete them by 'restart' and 'simulate'
        """

        results = {node.name: root_sequence} if node.name else {}
        print(type(results), results)
        for child in node.children:
            branch_length = float(child.dist)
            branch_step = int(branch_length * len(root_sequence))
            branch_step = max(1, branch_step)
            print(f"from {node.name} to {child.name}")
            print(f"Simluating {branch_step} steps on this branch")

            # restart the simulator
            self.restart(seq=root_sequence)
            # simulate the given steps
            self.simulate(step=branch_step)
            print("This branch is done!")
            simulated_sequence = self.updated_seq
            print(type(results), results)

            results.update(self.simulate_on_a_tree(child, simulated_sequence))

        return results

    def fake_simulate_on_a_tree(self, node, root_sequence):

        """
        simulate sequence evolution on a given phylogenetic tree - we just need to treat this task as separate
        independent 'simulate' processes and complete them by 'restart' and 'simulate'
        """

        results = {node.name: root_sequence} if node.name else {}
        print(type(results), results)
        for child in node.children:
            branch_length = float(child.dist)
            branch_step = int(branch_length * len(root_sequence))
            branch_step = max(1, branch_step)
            print(f"from {node.name} to {child.name}")
            print(f"Simluating {branch_step} steps on this branch")
            print("This branch is done!")
            simulated_sequence = self.updated_seq
            print(type(results), results)

            results.update(self.simulate_on_a_tree(child, simulated_sequence))

        return results







