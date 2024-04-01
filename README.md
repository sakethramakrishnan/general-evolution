# general-evolution

Clone the repository: 

```bash
git clone https://github.com/sakethramakrishnan/general-evolution.git
```

Go to the file ```recommend.py``` with:
```bash
cd general-evolution/evolution/
```

Look at the ```test.yaml``` file in: ```yaml/test.yaml```. We want our config files to look like that file. Here is a snippet of the config file with explanation:

```yaml
# genetic sequence we want to artificially mutate:
genetic_seq: 'ATGAAA'

model_paths:
  - model_1_path 
  - model_2_path 

# we currently accept BertForMaskedLM and GPTNeoXForCausalLM, but you can easily change these in the amis.py file
model_architectures:
  - 'neox'
  - 'neox'

# if you would like to train BPE tokenizers on the fly
train_path: training_data.fasta
validation_path: validation_data.fasta

# This is where it gets dense:
# We want each item in the list to be associated with the specific model
# I.e. we want model 1 to go with tokenizer 1, to go with vocab size 1, etc.
# Hence, the 1st item in tokenizer_paths, tokenizer_types, vocab_sizes, etc. should be connected with the 1st model path we listed
tokenizer_paths:
  - tokenizer_path_for_model_1 
  - tokenizer_path_for_model_2

# tokenizer types can be:
# dna_wordlevel, codon_wordlevel, protein_alphabet_wordlevel
# npe_tokenizer, cpe_tokenizer, ape_tokenizer
tokenizer_types:
  - tokenizer_type_for_model_1
  - tokenizer_type_for_model_2

# for BPE tokenizers; set equal to 0 for .json tokenizers
vocab_sizes:
  - 0
  - 0

convert_to_aa_list:
  - False
  - False


num_char_per_token_list:
  - 3
  - 3
```

Some things to note about the config .yaml file that you will be using:
- Each of the items in the list should be associated with their respective models and other items. I.e. the 1st item in each list should all be associated with the 1st model you trained
- EVEN IF A MODEL DOESN'T NEED AN INPUT PARAMETER FOR A SUBHEADING, INCLUDE A NULL VALUE ("", null, etc.) FOR IT. For instance, k-mer tokenizers do not need vocab_sizes, yet we still set them equal to 0. Essentially, we want each of the subheadings to have the same number of elements as the number of models you are observing.

After setting up the .yaml config file, which we will term ```test.yaml``` for this demonstration, run:
```bash
python recommend.py --config=test.yaml
```

The result will print something like:
```bash
A5T     3
C9A     3
C11T    3
A12C    3
```

Basically, the first line is saying:
- The current nucleotide in the genetic sequence you would like to mutate in the ```5th``` position is ```A```. 
- The models suggest that the nucleotide should be switched to ```T```.
- ```3``` models suggest this as a probable mutation.

This is the same interpretation for all the following lines.