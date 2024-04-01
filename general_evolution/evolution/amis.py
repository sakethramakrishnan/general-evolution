import numpy as np
import scipy.special
import sys
import torch
from transformers import (
    BertForMaskedLM,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerFast
    )
from pathlib import Path
from typing import Union, List
PathLike = Union[str, Path]
import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import trainers, processors, decoders
from tokenizers.pre_tokenizers import Whitespace, Digits
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset
import os


MODEL_DISPATCH = {
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "BertForMaskedLM": BertForMaskedLM,
    "neox": GPTNeoXForCausalLM,
    "bert": BertForMaskedLM,
    "GPT": GPTNeoXForCausalLM,
    "gpt": GPTNeoXForCausalLM
}

BPE_TOKENIZERS = ['ape_tokenizer', 'bpe_tokenizer', 'cpe_tokenizer', 'npe_tokenizer']

CODON_TO_CHAR = {
    "TCG": "A",
    "GCA": "B",
    "CTT": "C",
    "ATT": "D",
    "TTA": "E",
    "GGG": "F",
    "CGT": "G",
    "TAA": "H",
    "AAA": "I",
    "CTC": "J",
    "AGT": "K",
    "CCA": "L",
    "TGT": "M",
    "GCC": "N",
    "GTT": "O",
    "ATA": "P",
    "TAC": "Q",
    "TTT": "R",
    "TGC": "S",
    "CAC": "T",
    "ACG": "U",
    "CCC": "V",
    "ATC": "W",
    "CAT": "X",
    "AGA": "Y",
    "GAG": "Z",
    "GTG": "a",
    "GGT": "b",
    "GCT": "c",
    "TTC": "d",
    "AAC": "e",
    "TAT": "f",
    "GTA": "g",
    "CCG": "h",
    "ACA": "i",
    "CGA": "j",
    "TAG": "k",
    "CTG": "l",
    "GGA": "m",
    "ATG": "n",
    "TCT": "o",
    "CGG": "p",
    "GAT": "q",
    "ACC": "r",
    "GAC": "s",
    "GTC": "t",
    "TGG": "u",
    "CCT": "v",
    "GAA": "w",
    "TCA": "x",
    "CAA": "y",
    "AAT": "z",
    "ACT": "0",
    "GCG": "1",
    "GGC": "2",
    "CTA": "3",
    "AAG": "4",
    "AGG": "5",
    "CAG": "6",
    "AGC": "7",
    "CGC": "8",
    "TTG": "9",
    "TCC": "!",
    "TGA": "@",
    "XXX": "*",
}

CHAR_TO_CODON = {v: k for k, v in CODON_TO_CHAR.items()}

def read_fasta_only_seq(fasta_file: PathLike) -> List[str]:
    """Reads fasta file sequences without description tag."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace("\n", "") for seq in non_parsed_seqs for line in seq.split("\n", 1)
    ]

    return lines[1::2]


def cpe_decode(seq, *args, **kwargs) -> str:
        return "".join(CHAR_TO_CODON.get(c, "") for c in seq)

def any_file_fasta_reader(fasta_file: PathLike) -> List[str]:
    if Path(fasta_file).is_file():
        fasta_files = [fasta_file]
    else:
        fasta_files = Path(fasta_file).glob("*.fasta")

    sequences = []
    for p in fasta_files:
        sequences.extend(read_fasta_only_seq(p))

    return sequences

def group_and_contextualize(
    seq: str, num_char_per_token: int, convert_to_aa: bool, tokenizer_type: str
) -> str:
    """
    Prepares a sequence to be tokenized by the given tokenizer
    Note: all tokenizers require spaces between each character

    ape, npe, protein_alphabet, and dna_wordlevel should be k = 1
    cpe and codon_wordlevel should be k = 3

    Args:
        seq (str): one sequence of DNA nucleotides or amino acids
        k (int): the
        tokenizer_type (str): choices=['ape_tokenizer', 'npe_tokenizer', 'cpe_tokenizer', 'codon_wordlevel', 'dna_wordlevel', 'protein_alphabet_wordlevel']

    Returns:
        str: a string of the grouped, separated, and/or contextualized sequences
    """
    if tokenizer_type in ['npe_tokenizer', 'ape_tokenizer', 'cpe_tokenizer']:
        if tokenizer_type == "cpe_tokenizer":
            try:
                return "".join(
                    CODON_TO_CHAR[seq[i : i + num_char_per_token]]
                    for i in range(0, len(seq), num_char_per_token)
                )
            except KeyError:
                raise ValueError(f"Invalid sequence during codon to char:\n{seq}")
        elif tokenizer_type == 'npe_tokenizer':
            substrings = [
                seq[i : i + num_char_per_token]
                for i in range(0, len(seq), num_char_per_token)
            ]
        elif tokenizer_type == 'ape_tokenizer':
            substrings = [
                seq[i : i + num_char_per_token]
                for i in range(0, len(seq), num_char_per_token)
            ]
        else:
            raise ValueError(f"Invalid tokenizer type: {tokenizer_type}. Must be one of: ['ape_tokenizer', 'npe_tokenizer', 'cpe_tokenizer', 'codon_wordlevel', 'dna_wordlevel', 'protein_alphabet_wordlevel']")
        return "".join(substrings)

    else:
        if convert_to_aa:
            # this assumes you have the sequences already translated in a .fasta file
            substrings = [
                seq[i : i + num_char_per_token]
                for i in range(0, len(seq), num_char_per_token)
            ]
        else:  # Nucleotide case
            substrings = [
                seq[i : i + num_char_per_token]
                for i in range(0, len(seq), num_char_per_token)
            ]
        return " ".join(substrings)

def build_bpe_tokenizer(
    corpus_iterator: list[str], # consists of a list of genetic sequence to train the bpe tokenizer on
    vocab_size: int, # number of tokens in the vocabulary
    tokenizer_type: str, # choices: ['ape_tokenizer', 'npe_tokenizer', 'cpe_tokenizer']
    add_bos_eos: bool = True,
    save: bool = False,
    initial_alphabet: list[str] = None, # tokenizer must have these chars: ex. ['A', 'T', 'C', 'G'] for npe_tokenizer; uncomment the line with initial_alphabet
    save_name: str = '',
    cpe_translated: bool = False # whether or not the input corpus is already translated into CPE language
):
    special_tokens = {
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }

    bos_index = 5
    eos_index = 6

    # Define tokenizer
    tokenizer = Tokenizer(BPE(unk_token=special_tokens["unk_token"]))

    if tokenizer_type == "cpe_tokenizer":

        sequences = []
        for x in corpus_iterator:
            sequences.extend(x)
        try:
            corpus_iterator = [
                group_and_contextualize(
                    seq, num_char_per_token=3, convert_to_aa=False, tokenizer_type='cpe_tokenizer'
                )
                for seq in sequences
            ]

        except: # for when the sequences are already translated into CPE language
            pass

        initial_alphabet = list(CODON_TO_CHAR.values())

        tokenizer.pre_tokenizer = Digits(individual_digits=False)
    else:
        if tokenizer_type == 'ape_tokenizer':
            initial_alphabet = ['L','K', 'P','I','E','V','G','N','D','S','W','M','Y','R','C','A','T','F','_','Q','H','X']
        elif tokenizer_type == 'npe_tokenizer':
            initial_alphabet = ['A', 'T', 'C', 'G']


        tokenizer.pre_tokenizer = Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(special_tokens.values()),
        initial_alphabet=initial_alphabet # have an initial alphabet just in case of sequences without a particular base or amino acid :)
    )

    print("Training tokenizer")

    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    # Add post-processor
    # trim_offsets=True will ignore spaces, false will leave them in
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    if add_bos_eos:
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", bos_index), ("[EOS]", eos_index)],
        )

    # Add a decoder
    tokenizer.decoder = decoders.ByteLevel()

    # save the tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, **special_tokens
    )
    if save:

        wrapped_tokenizer.save_pretrained(save_name)

    print(f"Returning tokenizer with vocab_size = {tokenizer.get_vocab_size()}")

    return wrapped_tokenizer

class FastaDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        num_char_per_token: int,
        convert_to_aa: bool,
        tokenizer_type: str,
        # TODO: Play with filter function abstraction
        # filter_fnxs: Optional[List[Callable[[str], str]]] = None,
    ) -> None:
        # num_char_per_token is how many characters we tokenize
        # e.g. if our input_seq = 'AATTTGGGAATG' and convert_to_aa == False
        # Say we wanted to tokenize by codons; i.e. ['AAT', 'TTG', 'GGA', 'ATG']
        # then num_char_per_token = 3

        # Read the fasta file
        dna_sequences = any_file_fasta_reader(file_path)
        # Preprocess the sequences into codons
        # TODO: We could also use an <unk> token (this would be better)
        

        self.sequences = [
            group_and_contextualize(
                seq, num_char_per_token, convert_to_aa, tokenizer_type
            )
            for seq in dna_sequences
        ]
        
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        # Get the idx'th sequence
        return self.sequences[idx]

def err_model(name):
    raise ValueError('Model {} not supported'.format(name))

def get_model(args):
    if args.model_name == 'esm1b':
        from fb_model import FBModel
        model = FBModel(
            'esm1b_t33_650M_UR50S',
            repr_layer=[-1],
        )
    elif args.model_name.startswith('esm1v'):
        from fb_model import FBModel
        model = FBModel(
            'esm1v_t33_650M_UR90S_' + args.model_name[-1],
            repr_layer=[-1],
        )
    elif args.model_name == 'esm-msa':
        from fb_model import FBModel
        model = FBModel(
            'esm_msa1_t12_100M_UR50S',
            repr_layer=[-1],
        )
    elif args.model_name == 'prose':
        from prose_model import ProseModel
        model = ProseModel()
    else:
        err_model(args.model_name)

    return model

def get_model_name(name):
    if name == 'esm1b':
        from fb_model import FBModel
        model = FBModel(
            'esm1b_t33_650M_UR50S',
            repr_layer=[-1],
        )
    elif name.startswith('esm1v'):
        from fb_model import FBModel
        model = FBModel(
            'esm1v_t33_650M_UR90S_' + name[-1],
            repr_layer=[-1],
        )
    elif name == 'esm-msa':
        from fb_model import FBModel
        model = FBModel(
            'esm_msa1_t12_100M_UR50S',
            repr_layer=[-1],
        )
    elif name == 'prose':
        from prose_model import ProseModel
        model = ProseModel()
    else:
        err_model(name)

    return model

def get_model_path(model_path: str, model_architecture: str): 
    model = MODEL_DISPATCH[model_architecture].from_pretrained(
            model_path
        )
    
    return model

def get_tokenizer(tokenizer_path: str = '', tokenizer_type: str = '', vocab_size: int = 0, corpus_iterator = None):
    
    if os.path.isfile(Path(tokenizer_path)):
        # These are for the .json files
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrained_model_name_or_path=tokenizer_path
        )

    else:
        # These are for the bpe tokenizers

        if tokenizer_path not in ["", 'None', None]:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        else:
            
            # build bpe tokenizer on the fly
            tokenizer = build_bpe_tokenizer(
                    corpus_iterator = corpus_iterator,
                    vocab_size = vocab_size,
                    tokenizer_type = tokenizer_type,
                    save = False
                    )
    
    special_tokens = {
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
        }

    # for some reason, we need to add the special tokens even though they are in the json file
    tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer


def encode(seq, model, tokenizer, num_char_per_token, convert_to_aa, tokenizer_type, max_position_emb):
    grouped_seq = group_and_contextualize(seq, num_char_per_token, convert_to_aa, tokenizer_type)

    
    input = tokenizer(
            grouped_seq,
            return_tensors="pt",
            truncation=True,
            padding='longest',
            max_length=max_position_emb,
        )

    
    outputs = model(input_ids = input["input_ids"],
            attention_mask = input["attention_mask"],
            output_hidden_states=True,)
    
    logits = outputs.logits
    # logits in shape: (batch_size, num_tokens, vocab_size)
    
    return logits

def decode(logits, model, tokenizer, exclude=set()):
    # I'll be completely honest, I have no idea why the original authors used the alphabet and stuff
    logits = torch.squeeze(logits, dim = 0).detach().numpy() # We want to remove the first dimension because there is only 1 sequence, so batch size = 1

    tokenizer_vocab = list(tokenizer.vocab.keys())
    

    assert(logits.shape[1] == len(tokenizer_vocab))

    valid_idx = [
        idx for idx, tok in enumerate(tokenizer_vocab)
    ]
    
    #print(valid_idx)
    
    logits = logits[:, valid_idx]
    
    argmax_indices = np.argmax(logits, axis=-1)
    argmax_seq = tokenizer.decode(argmax_indices)

    # NOTE: Below is the original implementation
    # argmax = ''.join([
    #     tokenizer_vocab[valid_idx[tok_idx]]
    #     if ('<' not in tokenizer_vocab[valid_idx[tok_idx]] and
    #         tokenizer_vocab[valid_idx[tok_idx]] not in exclude) else '.'
    #     for tok_idx in np.argmax(logits, 1)
    # ])

    return argmax_seq

def deep_mutational_scan(seq, model):
    if model.name_ == 'prose':
        from prose.alphabets import Uniprot21
        from prose.utils import pack_sequences
        alphabet = Uniprot21()
        x = [ torch.from_numpy(alphabet.encode(seq.encode())).long() ]
        x, _ = pack_sequences(x)
        logits = model.model_(x).data.cpu().detach().numpy()

        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                pos, mt = i + 1, alphabet.decode(j).decode('utf-8')
                val = logits[i, j]
                print(f'{pos}\t{mt}\t{val}')
    else:
        logits = model.decode(model.encode(seq))

        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                pos, mt = i + 1, model.alphabet_.all_toks[j]
                val = logits[i, j]
                print(f'{pos}\t{mt}\t{val}')

def reconstruct(seq, model, tokenizer, num_char_per_token, convert_to_aa, tokenizer_type, max_position_emb: int, encode_kwargs={}, decode_kwargs={}):
    # if model.name_ == 'prose':
    #     return reconstruct_prose(seq, model)
    
    return_thing = decode(
        encode(seq, model, tokenizer, num_char_per_token, convert_to_aa, tokenizer_type, max_position_emb, **encode_kwargs),
        model, tokenizer, **decode_kwargs
    )
    
    return return_thing

def soft_reconstruct(seq, model, alpha=1., offset=1):
    if model.name_ == 'prose':
        raise NotImplementedError('Does not support prose reconstruction')

    exclude = set([
        'B', 'J', 'O', 'U', 'X', 'Z', '-', '.',
    ])

    logits = model.predict_sequence_prob(seq)
    probs = scipy.special.softmax(logits, axis=1)

    mutations = []
    
    for i in range(probs.shape[0] - 1):
        if i == 0:
            continue
        pos = i - offset
        wt_j = model.alphabet_.tok_to_idx[seq[pos]]
        wt_prob = probs[i, wt_j]
        for j in range(probs.shape[1]):
            mt = model.alphabet_.all_toks[j]
            if mt in exclude or '<' in mt:
                continue
            if j == wt_j:
                continue
            mt_prob = probs[i, j]
            if mt_prob > alpha * wt_prob:
                mutations.append((pos, seq[pos], mt))

    return mutations

def reconstruct_prose(seq, model):
    from prose.alphabets import Uniprot21
    from prose.utils import pack_sequences
    
    alphabet = Uniprot21()
    x = [ torch.from_numpy(alphabet.encode(seq.encode())).long() ]
    x, _ = pack_sequences(x)
    
    logits = model.model_(x).data.cpu().detach().numpy()

    return ''.join([
        alphabet.decode(np.argmax(logits[i])).decode('utf-8')
        for i in range(logits.shape[0])
    ])

def compare(seq_old, seq_new, start=0, end=None, namespace=None):
    #print(f'Old: {seq_old}')
    #print(f'New: {seq_new}')
    if namespace is not None:
        sys.stdout.write(f'{namespace} mutations: ')
    for idx, (ch_old, ch_new) in enumerate(
            zip(seq_old, seq_new)
    ):
        if idx < start:
            continue
        if end is not None and idx >= end:
            continue
        if ch_new == '.':
            continue
        if ch_old != ch_new:
            sys.stdout.write(f'{ch_old}{idx - start + 1}{ch_new}, ')
    sys.stdout.write('\n')

def diff(seq_old, seq_new, start=0, end=None):
    different_muts = []
    for idx, (ch_old, ch_new) in enumerate(
            zip(seq_old, seq_new)
    ):
        if idx < start:
            continue
        if end is not None and idx >= end:
            continue
        if ch_new == '.':
            continue
        if ch_old != ch_new:
            different_muts.append((idx, ch_old, ch_new))
    return different_muts
    
    
def reconstruct_multi_models(
        wt_seq,
        model_paths=[],
        model_architectures: list[str] = [],
        tokenizer_paths: list[str] = [],
        tokenizer_types: list[str] = [],
        vocab_sizes: list[int] = [],
        convert_to_aa_list: list[bool] = [],
        num_char_per_token_list: list[int] = [],
        train_path: str = '',
        validation_path: str = '',
        alpha=None,
        return_names=False,
):
    
    mutations_models, mutations_model_names = {}, {}
    
    for model_path, model_arch, tokenizer_path, tokenizer_type, vocab_size, convert_to_aa, num_char_per_token in zip(
    model_paths, 
    model_architectures, 
    tokenizer_paths, 
    tokenizer_types, 
    vocab_sizes, 
    convert_to_aa_list, 
    num_char_per_token_list): 
        
        model = get_model_path(model_path, model_arch)
        
        max_position_emb = model.config.max_position_embeddings
        
        train_dataset = FastaDataset(
            train_path,
            num_char_per_token=num_char_per_token,
            convert_to_aa=convert_to_aa,
            tokenizer_type=tokenizer_type,
        )
        eval_dataset = FastaDataset(
            validation_path,
            num_char_per_token=num_char_per_token,
            convert_to_aa=convert_to_aa,
            tokenizer_type=tokenizer_type,
        )
        
        full_dataset = train_dataset + eval_dataset
        
        tokenizer = get_tokenizer(tokenizer_path, tokenizer_type, vocab_size, full_dataset)
        
        
        if alpha is None:
            wt_new = reconstruct(
                wt_seq, model, tokenizer, num_char_per_token, convert_to_aa, tokenizer_type, max_position_emb, decode_kwargs={ 'exclude': 'unnatural' }
            )
            
            if tokenizer_type == 'cpe_tokenizer':
                wt_new = cpe_decode(wt_new)
                
            wt_new = wt_new.replace(" ", "")
            
            mutations_model = diff(wt_seq, wt_new)

        else:
            mutations_model = soft_reconstruct(
                wt_seq, model, alpha=alpha,
            )
            
        model_name = model_arch + '_' + tokenizer_type + '_' + str(vocab_size)

        # mutations_model: list of tuples of:
        # (idx, 'base_nucleotide', 'mutated_nucleotide')

        for mutation in mutations_model:
            if mutation not in mutations_models:
                mutations_models[mutation] = 0
                mutations_model_names[mutation] = []
            mutations_models[mutation] += 1
            mutations_model_names[mutation].append(model_name)
        del model

    if return_names:
        return mutations_models, mutations_model_names

    return mutations_models

def evolve(seq_uca, model, n_generations=1):
    print(f'Gen 0: {seq_uca}')
    seq_curr = seq_uca
    for gen in range(n_generations):
        seq_ml = reconstruct(seq_curr, model)
        compare(seq_curr, seq_ml)
        if seq_ml == seq_curr:
            print(f'Converged at generation {gen}')
            break
        print(f'Gen {gen + 1}: {seq_ml}')
        seq_curr = seq_ml

def interpolate(baseline, target, model, n_steps=15,
                start=0, end=None):
    for alpha in np.linspace(0., 1., n_steps):
        midpoint = (alpha * encode(baseline, model)) + \
                   ((1 - alpha) * encode(target, model))
        new_seq = decode(midpoint, model)
        compare(target, new_seq, start, end)

def extrapolate(baseline, target, model, n_steps=15,
                start=0, end=None):
    for alpha in np.linspace(1., 5., n_steps):
        delta = (encode(target, model) - encode(baseline, model))
        new_seq = decode((delta * alpha) + encode(target, model), model)
        compare(target, new_seq, start, end)


def select_top_n(numbers, n):
    sorted_numbers = sorted(numbers, reverse=True)
    top_n = sorted_numbers[:n]
    return top_n

def top_n_occurrences(data: list[tuple], n):
    occurence_of_num_models = [] # num_models: occurrence
    
    for tup in data:
        models = tup[1]
        if models not in occurence_of_num_models:
            occurence_of_num_models.append(models)
        
    top_n_occurrences = select_top_n(occurence_of_num_models, n)
    return_muts = []
    
    for tup in data:
        models = tup[1]
        if models in top_n_occurrences:
            return_muts.append(tup)
    
    return return_muts
