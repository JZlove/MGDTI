import numpy as np
import pandas as pd
import os

class Prepare():
    def __init__(self):
        self.dg_dg_path = 'origin_dataset/drug_drug.csv'
        self.dg_pt_path = 'origin_dataset/drug_protein.csv'
        self.pt_pt_path = 'origin_dataset/protein_protein.csv'
        self.dg_dg = pd.read_csv(self.dg_dg_path, header=0, index_col=0).values
        self.dg_pt = pd.read_csv(self.dg_pt_path, header=0, index_col=0).values
        self.pt_pt = pd.read_csv(self.pt_pt_path, header=0, index_col=0).values
        self.num_dg = self.dg_dg.shape[0]
        self.num_pt = self.pt_pt.shape[0]

    def prepare_fold(self, dg_pt, node_list, node_type, delete_rate):
        whole_positive_index = []
        whole_negetive_index = []

        for i in range(dg_pt.shape[0]):
            for j in range(dg_pt.shape[1]):
                if node_type == 0 and i not in node_list:
                    continue
                if node_type == 1 and j not in node_list:
                    continue

                if dg_pt[i][j] == 1:
                    whole_positive_index.append([i, j, 1])
                else:
                    whole_negetive_index.append([i, j, 0])
        positive_num = len(whole_positive_index)
        select_positive_num = int(positive_num*delete_rate)
        positive_index = [i for i in range(positive_num)]
        np.random.shuffle(positive_index)            
        negtive_index = np.random.choice(np.arange(len(whole_negetive_index)),
                                            size=positive_num, replace=False)
        
        fold_index_sel = []
        fold_index_res = []
        for index in range(positive_num):
            if index < select_positive_num:
                fold_index_sel.append(whole_positive_index[positive_index[index]])
                fold_index_sel.append(whole_negetive_index[negtive_index[index]])
            else:
                fold_index_res.append(whole_positive_index[positive_index[index]])
                fold_index_res.append(whole_negetive_index[negtive_index[index]])
        return fold_index_sel, fold_index_res

    def build_fold_cold_drug(self, fold_num, cold_drug_list, train_drug_list, delete_rate):
        path = f"cold_dataset/cold_drug/delete_rate_{delete_rate}"
        if not os.path.exists(path):
            os.mkdir(path)

        dg_dg = self.dg_dg.copy()
        dg_pt = self.dg_pt.copy()
        pt_pt = self.pt_pt.copy()
        fold_train_index = []
        fold_test_index, fold_train_index_ = self.prepare_fold(dg_pt, cold_drug_list, 0, delete_rate)
        fold_train_index.extend(fold_train_index_)
        fold_train_index_, _ = self.prepare_fold(dg_pt, train_drug_list, 0, 1.0)
        fold_train_index.extend(fold_train_index_)

        for test_index in fold_test_index:
            d_id, t_id, _ = test_index
            dg_pt[d_id][t_id] = 0

        for drug_id in cold_drug_list:
            all_edges = np.where(dg_dg[drug_id] == 1)[0]
            np.random.shuffle(all_edges)
            num_edges = len(all_edges)
            delete_num = int(delete_rate*num_edges)
            res_edges = all_edges[delete_num:]
            dg_dg[drug_id] = 0
            for edge in res_edges:
                dg_dg[drug_id][edge] = 1
        
        
        np.save(os.path.join(path, f"fold_{fold_num}_dg_dg.npy"), dg_dg)
        np.save(os.path.join(path, f"fold_{fold_num}_pt_pt.npy"), pt_pt)
        np.save(os.path.join(path, f"fold_{fold_num}_dg_pt.npy"), dg_pt)
        np.save(os.path.join(path, f"fold_{fold_num}_test_index.npy"), fold_test_index)
        np.save(os.path.join(path, f"fold_{fold_num}_train_index.npy"), fold_train_index)


    def build_cold_drug(self, delete_rate):
        fold_nums = 10
        drug_list = [i for i in range(self.num_dg)]
        np.random.shuffle(drug_list)
        split_len = int(self.num_dg // fold_nums)
        cnt = 0
        for fold_num in range(fold_nums):
            if fold_num < fold_nums - 1:
                cold_drug_list = drug_list[cnt: cnt+split_len]
                train_drug_list = drug_list[0:cnt] + drug_list[cnt+split_len: ]
            else:
                cold_drug_list = drug_list[cnt:]
                train_drug_list = drug_list[:cnt]
            self.build_fold_cold_drug(fold_num, cold_drug_list, train_drug_list, delete_rate)
            cnt += split_len


    def build_fold_cold_protein(self, fold_num, cold_protein_list, train_protein_list, delete_rate):
        path = f"cold_dataset/cold_protein/delete_rate_{delete_rate}"
        if not os.path.exists(path):
            os.mkdir(path)

        dg_dg = self.dg_dg.copy()
        dg_pt = self.dg_pt.copy()
        pt_pt = self.pt_pt.copy()
        fold_train_index = []
        fold_test_index, fold_train_index_ = self.prepare_fold(dg_pt, cold_protein_list, 1, delete_rate)
        fold_train_index.extend(fold_train_index_)
        fold_train_index_, _ = self.prepare_fold(dg_pt, train_protein_list, 1, 1.0)
        fold_train_index.extend(fold_train_index_)

        for test_index in fold_test_index:
            d_id, t_id, _ = test_index
            dg_pt[d_id][t_id] = 0

        for protein_id in cold_protein_list:
            all_edges = np.where(pt_pt[protein_id] == 1)[0]
            np.random.shuffle(all_edges)
            num_edges = len(all_edges)
            delete_num = int(delete_rate*num_edges)
            res_edges = all_edges[delete_num:]
            pt_pt[protein_id] = 0
            for edge in res_edges:
                pt_pt[protein_id][edge] = 1
        
        
        np.save(os.path.join(path, f"fold_{fold_num}_dg_dg.npy"), dg_dg)
        np.save(os.path.join(path, f"fold_{fold_num}_pt_pt.npy"), pt_pt)
        np.save(os.path.join(path, f"fold_{fold_num}_dg_pt.npy"), dg_pt)
        np.save(os.path.join(path, f"fold_{fold_num}_test_index.npy"), fold_test_index)
        np.save(os.path.join(path, f"fold_{fold_num}_train_index.npy"), fold_train_index)

    def build_cold_protein(self, delete_rate):
        fold_nums = 10
        protein_list = [i for i in range(self.num_pt)]
        np.random.shuffle(protein_list)
        split_len = int(self.num_pt // fold_nums)
        cnt = 0
        for fold_num in range(fold_nums):
            if fold_num < fold_nums - 1:
                cold_protein_list = protein_list[cnt: cnt+split_len]
                train_protein_list = protein_list[0:cnt] + protein_list[cnt+split_len: ]
            else:
                cold_protein_list = protein_list[cnt:]
                train_protein_list = protein_list[:cnt]
            self.build_fold_cold_protein(fold_num, cold_protein_list, train_protein_list, delete_rate)
            cnt += split_len