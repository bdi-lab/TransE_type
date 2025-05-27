Copyright: KAIST Big Data Intelligence Lab (https://bdi-lab.kaist.ac.kr/)

Knowledge Graph Embedding with Entity Type Constraints, Seunghwan Kong, Chanyoung Chung, Suheon Ju, and Joyce Jiyoung Whang, Journal of KIISE, Sep. 2022

All codes are written by Seunghwan Kong ([shkong@kaist.ac.kr](mailto:shkong@kaist.ac.kr)) and proofread by Chanyoung Chung ([chanyoung.chung@kaist.ac.kr](mailto:chanyoung.chung@kaist.ac.kr)).

### Data

To run the code, the dataset should be located in "./data/[DATASET_NAME]" directory with the following files:

- entity2id.txt: Each line is in the form of "entity entity_index". The first line indicates the number of entities.
- entity2typeid.txt: Each line is in the form of "entity_index type_index".
- relation2id.txt: Each line is in the form of "relation relation_index". The first line indicates the number of relations.
- train2id.txt / valid2id.txt / test2id.txt: Each line is in the form of "head_entity_index, tail_entity_index, relation_index". First line indicates the number of triplets.
- triplets.txt: Each line is in the form of "head_entity relation tail_entity". The first line indicates the number of triplets.
- type2id.txt: Each line is in the form of "type type_index". The first line indicates the number of types.

### train_transe_type.py

Code for generating knowledge graph embeddings with entity type constraints.

**Arguments**

- neg: Number of negative samples to generate for each triplet (25 as default)
- dim: Dimension of the embeddings (128 as default)
- epochs: Total epochs for experiment (1000 as default)
- valid_epochs: Number of epochs between validations (50 as default)
- lamb: Weight of type loss (0.01 as default)
- data: Name of the dataset (None as default)
- lr: Learning rate (2.0 as default)
- margin: Margin of the margin ranking loss (1.0 as default)
- test: 0 for the validation 1 for the test (0 as default)

**Usage:** python3 train_transe_type.py [arguments]

- Example: python3 train_transe_type.py -lamb 0.05 -data [DATASET_NAME] -lr 0.1

**Output**

- Mean rank, mean reciprocal rank, hit@1,3,10 of the trained model in every validation epoch

### License

Our codes are released under the CC BY-NC-SA 4.0 license.
