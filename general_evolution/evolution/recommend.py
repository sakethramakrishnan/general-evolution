import os
from amis import reconstruct_multi_models, top_n_occurrences
import os
from argparse import ArgumentParser
from dataclasses import dataclass, field

import yaml


@dataclass
class EvolutionParams:
    genetic_seq: str = ""
    
    alpha: float = None # alpha stringency parameter
    
    model_paths: list[str] = field(default_factory=list)
    model_architectures: list[str] = field(default_factory=list)
    max_length: int = 1080
    
    
    train_path: str = ""
    validation_path: str = ""
    
    tokenizer_paths: list[str] = field(default_factory=list)
    tokenizer_types: list[str] = field(default_factory=list)
    vocab_sizes: list[int] = field(default_factory=list)
    convert_to_aa_list: list[bool] = field(default_factory=list)
    num_char_per_token_list: list[int] = field(default_factory=list)  # how many characters per token

    # def __post_init__(self):

    #     # Configure tokenization parameters
    #     if self.tokenizer_type in ["ape_tokenizer", "protein_alphabet_wordlevel"]:
    #         self.convert_to_aa = True
    #         self.num_char_per_token = 1
    #     elif self.tokenizer_type in ["npe_tokenizer", "dna_wordlevel"]:
    #         self.convert_to_aa = False
    #         self.num_char_per_token = 1
    #     elif self.tokenizer_type in ["cpe_tokenizer", "codon_wordlevel"]:
    #         self.convert_to_aa = False
    #         self.num_char_per_token = 3
    #     else:
    #         raise ValueError(f"Invalid tokenizer_type: {self.tokenizer_type}")

    #     # for some reason, the default mlm_probability is a tuple
    #     if type(self.mlm_probability) == tuple:
    #         self.mlm_probability = float(self.mlm_probability[0])
        
    #     # Log the config to a yaml file
    #     with open(os.path.join(self.output_dir, "train_config.yaml"), "w") as fp:
    #         yaml.dump(asdict(self), fp)



def parse_args():

    parser = ArgumentParser(
        description='Recommend substitutions to a wildtype sequence'
    )
    
    parser.add_argument("--config", type=str, default='/home/couchbucks/Documents/saketh/cpe-evolution/bin/evolution/yaml/test.yaml')
    
    parser.add_argument(
        '--cuda',
        type=str,
        default='cuda',
        help='cuda device to use'
    )
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    #sequence = 'QVQLQQSGPGLVKPSQTLSLTCAISGDSVSSYNAVWNWIRQSPSRGLEWLGRTYYRSGWYNDYAESVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCARSGHITVFGVNVDAFDMWGQGTMVTVSS'
    args = parse_args()
    with open(args.config) as fp:
        config = EvolutionParams(**yaml.safe_load(fp))
        
    if ":" in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda.split(':')[-1]
    
    mutations_models = reconstruct_multi_models(
        config.genetic_seq,
        config.model_paths,
        config.model_architectures,
        config.tokenizer_paths,
        config.tokenizer_types,
        config.vocab_sizes, 
        config.convert_to_aa_list, 
        config.num_char_per_token_list,
        config.train_path,
        config.validation_path,
        config.alpha,
    )
    
    mutations_list = [(mut, mutations_models[mut]) for mut in mutations_models]
    
    
    top_mutations = top_n_occurrences(mutations_list, n=1)

    # Print the top mutations
    sorted_data = sorted(top_mutations, key=lambda x: x[1], reverse=True)

    print("")
    print("The mutations are in this format:")
    print('{original nucleotide}{idx of nucleotide}{proposed mutation}    {number of models that proposed it}')

    genetic_list = list(config.genetic_seq)
    
    
    for item in sorted_data:
        element, count = item
        idx = element[0]
        original = element[1]
        proposed = element[2]
        print(f"{original}{idx}{proposed}\t{count}")
        
        # assert genetic_list[idx] == original, f"original: {original}   mutated: {genetic_list[idx]}"
        genetic_list[idx] = proposed
    
    print("")
    print("Proposed mutated string:")
    print(f"{''.join(genetic_list)}")

    # for k, v in sorted(mutations_models.items(), key=lambda item: -item[1]):
    #     mut_str = f'{k[1]}{k[0] + 1}{k[2]}'
    #     print(f'{mut_str}\t{v}')
