import os
import random
import dgl
import numpy as np
import torch
import torch.nn.functional as F
from config import Config
from model import mainModel
from dataset import Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

os.environ['CUDA'] = '1'
cuda = "cuda:1"
device = torch.device(cuda)
config = Config()

def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def to_floattensor(x, use_gpu=True):
    if type(x) == 'tensor':
        if x.is_cuda:
            x = torch.cuda.FloatTensor(x)
        else:
            x = torch.FloatTensor(x)
            if use_gpu:
                x = x.cuda(cuda)
    else:
        x = torch.FloatTensor(x)
        if use_gpu:
            x = x.cuda(cuda)
    return x

def get_seqs(g, all_nodes):
    seq_len = config.seq_len
    n = 0
    node_seq = torch.zeros(len(all_nodes), seq_len).long()
    for x in all_nodes:

        cnt = 0
        scnt = 0
        node_seq[n, cnt] = x
        cnt += 1
        start = node_seq[n, scnt].item()
        while (cnt < seq_len):
            sample_list = g.successors(start).numpy().tolist()
            if len(sample_list) == 0:
                sample_list.append(start)
            nsampled = max(len(sample_list), 1)
            sampled_list = random.sample(sample_list, nsampled)
            for i in range(nsampled):
                node_seq[n, cnt] = sampled_list[i]
                cnt += 1
                if cnt == seq_len:
                    break
            scnt += 1
            start = node_seq[n, scnt].item()
        n += 1
    return node_seq

def get_metric(tag, pred):
    auc = roc_auc_score(tag, pred)
    aupr = average_precision_score(tag, pred)
    pred = [1 if x > 0.5 else 0 for x in pred]
    acc = accuracy_score(tag, pred)
    return auc, aupr, acc

def ramdom_sample(dg_train, pt_train, result_train):
    num = len(dg_train)
    index = [i for i in range(num)]
    np.random.shuffle(index)
    split_len = int(0.1*num)
    support_dg = dg_train[index[:split_len]]
    support_pt = pt_train[index[:split_len]]
    support_y = result_train[index[:split_len]]
    query_dg = dg_train[index[split_len:]]
    query_pt = pt_train[index[split_len:]]
    query_y = result_train[index[split_len:]]
    return support_dg, support_pt, support_y, query_dg, query_pt, query_y

def train_model(net, optimizer, fold_nums, epoch, features_list, node_type, type_emb, dg_seq, pt_seq, val):
    net.train()
    val = to_floattensor(val)
    support_dg, support_pt, support_y, query_dg, query_pt, query_y = ramdom_sample(dg_seq, pt_seq, val)

    ml_weights = net.update_parameters()
    support_pred = net(features_list, support_dg, support_pt, type_emb, node_type, ml_weights)
    support_loss = F.binary_cross_entropy(support_pred, support_y)
    grads = torch.autograd.grad(support_loss, ml_weights.values())
    gvs = dict(zip(ml_weights.keys(), grads))
    fast_weights = dict(
        zip(ml_weights.keys(), [ml_weights[key] - 1e-4*gvs[key] for key in ml_weights.keys()])
    )    
    query_pred =  net(features_list, query_dg, query_pt, type_emb, node_type, fast_weights)
    query_loss = F.binary_cross_entropy(query_pred, query_y)
    query_loss_list = []
    query_loss_list.append(query_loss)

    for _ in range(1, config.meta_k):
        net.train()
        support_pred = net(features_list, support_dg, support_pt, type_emb, node_type, fast_weights)
        support_loss = F.binary_cross_entropy(support_pred, support_y)
        grads = torch.autograd.grad(support_loss, fast_weights.values())
        gvs = dict(zip(fast_weights.keys(), grads))
        fast_weights = dict(
            zip(fast_weights.keys(), [fast_weights[key] - 1e-4*gvs[key] for key in fast_weights.keys()]))

        query_pred = net(features_list, query_dg, query_pt, type_emb, node_type, fast_weights)
        query_loss = F.binary_cross_entropy(query_pred, query_y)
        query_loss_list.append(query_loss)
    
    loss = torch.stack(query_loss_list).mean(0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(' Fold_num {:02d} | Epoch {:03d} | Train_Loss: {:.4f}'.format(fold_nums,
                                                                        epoch, loss.item()))

def vaild_model(net, features_list, node_type, type_emb, dg_seq, pt_seq, val):
    net.eval()
    with torch.no_grad():
        pred = net(features_list, dg_seq, pt_seq, type_emb, node_type)
        pred = pred.cpu().numpy()
    auc, aupr, acc = get_metric(val, pred)
    print(f"valid: auc: {auc} aupr: {aupr} acc: {acc} ")
    return auc, aupr, acc

def run():
    
    config = Config()
    dataset = Dataset()
    num_dg = dataset.num_dg
    num_pt = dataset.num_pt

    delete_rate_list = [1.0]
    del_auc_list = []
    del_aupr_list = []

    for delete_rate in delete_rate_list:
        acc_list = []
        auc_list = []
        aupr_list = []
        for fold_num in range(config.fold_nums):
            print(f"crossfold: {fold_num+1} ++++++++++++++++++++++")
            features_list, adjM, train_set, test_set = dataset.get_fold_data(fold_num, delete_rate)
            features_list = [mat2tensor(features).to(device)
                            for features in features_list]
            node_cnt = [features.shape[0] for features in features_list]
            sum_node = 0
            for x in node_cnt:
                sum_node += x

            in_dims = [features.shape[0] for features in features_list]
            for i in range(len(features_list)):
                dim = features_list[i].shape[0]
                indices = np.vstack((np.arange(dim), np.arange(dim)))
                indices = torch.LongTensor(indices)
                values = torch.FloatTensor(np.ones(dim))
                features_list[i] = torch.sparse.FloatTensor(
                    indices, values, torch.Size([dim, dim])).to(device)

            g = dgl.DGLGraph(adjM+(adjM.T))
            g = dgl.remove_self_loop(g)

            node_type = [i for i, z in zip(
                range(len(node_cnt)), node_cnt) for x in range(z)]
            
            type_emb = torch.eye(len(node_cnt)).to(device)
            node_type = torch.tensor(node_type).to(device)

            node_dg_index_train = train_set[:, 0]
            node_pt_index_train = train_set[:, 1]

            node_dg_index_test = test_set[:, 0]
            node_pt_index_test = test_set[:, 1]

            all_dg_node = [i for i in range(num_dg)]
            all_pt_node = [i for i in range(num_dg, num_dg+num_pt)]
            all_dg_seq = get_seqs(g, all_dg_node)
            all_pt_seq = get_seqs(g, all_pt_node)

            g = g.to(device)
            net = mainModel(g, in_dims, config.embedding_size, config.num_layers, config.num_gnns,
                            config.num_heads, config.dropout, temper=config.temper, num_type=len(node_cnt), beta=config.beta)
            net = net.to(device)
            # print(net.parameters)
            optimizer = torch.optim.Adam(
                net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min')
            
            best_auc = 0
            best_aupr = 0
            best_acc = 0
            dg_seq_test = all_dg_seq[node_dg_index_test]
            pt_seq_test = all_pt_seq[node_pt_index_test - num_dg]
            dg_seq_train = all_dg_seq[node_dg_index_train]
            pt_seq_train = all_pt_seq[node_pt_index_train - num_dg]

            for epoch in range(config.num_epochs):
                train_model(net, optimizer, fold_num, epoch, features_list,
                            node_type, type_emb, dg_seq_train, pt_seq_train, train_set[:, 2])
                auc, aupr, acc = vaild_model(net, features_list, node_type, type_emb,
                                dg_seq_test, pt_seq_test, test_set[:, 2])
                if auc > best_auc:
                    best_auc = auc
                    best_aupr = aupr
                    best_acc = acc
            auc_list.append(best_auc)
            aupr_list.append(best_aupr)
            acc_list.append(best_acc)
            print(f"best auc: {best_auc} best aupr: {best_aupr} best acc: {best_acc}")
        print(f"delete_rate: {delete_rate}")
        for i in range(config.fold_nums):
            print(f"fold {i} acc: {acc_list[i]} auc: {auc_list[i]} aupr: {aupr_list[i]}")
        print(f"avg_acc: {np.mean(acc_list)}")
        print(f"avg_auc: {np.mean(auc_list)}")
        print(f"avg_aupr: {np.mean(aupr_list)}")
        del_auc_list.append(np.mean(auc_list))
        del_aupr_list.append(np.mean(aupr_list))

    for index in range(len(del_auc_list)):
        print(f"delete_rate: {delete_rate_list[index]} auc: {del_auc_list[index]} aupr: {del_aupr_list[index]}")


if __name__ == "__main__":
    config = Config()
    dgl.random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run()
