from configparser import ConfigParser

class Config():
    def __init__(self):
        self.use_gpu = True
        self.gpu = 1
        self.result_path = 'result'
        self.seed = 2000213
        self.seq_len = 10
        self.meta_k = 5
        self.hidden_dim = 256
        self.num_layers = 2
        self.num_gnns = 3
        self.num_heads = 2
        self.dropout = 0.2
        self.temper = 1.0
        self.beta = 0.2
        self.lr = 2e-4
        self.weight_decay = 0
        self.embedding_size = 256
        self.fold_nums = 10
        self.num_epochs = 200

        self.dg_ds_path = 'origin_dataset/drug_disease.csv'
        self.dg_dg_path = 'origin_dataset/drug_drug.csv'
        self.dg_pt_path = 'origin_dataset/drug_protein.csv'
        self.dg_se_path = 'origin_dataset/drug_se.csv'
        self.pt_ds_path = 'origin_dataset/protein_disease.csv'
        self.pt_pt_path = 'origin_dataset/protein_protein.csv'
        self.dg_dg_sim_path = "origin_dataset/Similarity_Matrix_Drugs.txt"
        self.pt_pt_sim_path = "origin_dataset/Similarity_Matrix_Proteins.txt"
        self.dg_smiles_path = 'origin_dataset/drug_smiles.csv'
        self.pt_fasta_path = 'origin_dataset/protein_fasta.csv'

        self.cold_drug_path = 'cold_dataset/cold_drug'
        self.cold_protein_path = 'cold_dataset/cold_protein'