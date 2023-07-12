# -------------------------------------- ArgParse -------------------------------------- #
import argparse

import numpy as np
import torch

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1998, help='Random seed.')
parser.add_argument('--model', type=str, default='GraphSage', help='model', choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
parser.add_argument('--dataset', type=str, default='Cora', help='Dataset', choices=['Cora', 'Citeseer', 'Pubmed', 'Flickr', 'ogbn-arxiv', 'Reddit2'])
parser.add_argument('--train_lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5, help='Threshold')
parser.add_argument('--target_class', type=int, default=0, help='fake label')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int, default=1, help='Number of inner')
parser.add_argument('--debug', type=bool, default=True)

# backdoor setting
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3, help='trigger_size')
parser.add_argument('--use_vs_number', default=True, help="if use detailed number to decide Vs")
parser.add_argument('--vs_number', type=int, default=80, help="number of poisoning nodes relative to the full graph")
parser.add_argument('--vs_ratio', type=float, default=0, help="ratio of poisoning nodes relative to the full graph")

# defense setting
parser.add_argument('--defense_mode', type=str, default="none", choices=['prune', 'isolate', 'none'], help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.8, help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1, help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=100, help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.8, help="Threshold of increase similarity")

# attack setting
parser.add_argument('--dis_weight', type=float, default=1, help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='cluster_degree', choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'], help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN', choices=['GCN', 'GAT', 'GraphSage', 'GIN'], help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='1by1', choices=['overall', '1by1'], help='Model used to attack')

# GPU setting
parser.add_argument('--device_id', type=int, default=0)

# args process
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


# -------------------------------------- dataset -------------------------------------- #
# inductive learning, 训练时仅采用训练集节点之间的连边, 不包含训练集和测试集(验证集)连边
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit2, Flickr
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset

from utils import get_split
from utils import subgraph

assert args.dataset is not None, 'choose the dataset!!!'
dataset = None
transform = T.Compose([T.NormalizeFeatures()])
if args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed':
    dataset = Planetoid(root='./data/', name=args.dataset, transform=transform)
elif args.dataset == 'Flickr':
    dataset = Flickr(root='./data/Flickr/', transform=transform)
elif args.dataset == 'Reddit2':
    dataset = Reddit2(root='./data/Reddit2/', transform=transform)
elif args.dataset == 'ogbn-arxiv':
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')

data = dataset[0].to(device=device.type)

if args.dataset == 'ogbn-arxiv':
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)

#  train_set: 20%, val_set: 10%, clean_test_set: 10%, dirty_test_set: 10%
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args, data, device)
data.edge_index = to_undirected(data.edge_index)  # only support undirected graph

# 不包含测试集节点的边集
not_test_edge_index, not_test_edge_mask = subgraph(subset=torch.bitwise_not(data.test_mask), edge_index=data.edge_index)

# 只包含测试集节点的连边集合
test_edge_index = data.edge_index[:, torch.bitwise_not(not_test_edge_mask)]

# 非测试集非验证集部分的节点 idx
unlabeled_idx = (torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.val_mask)).nonzero().flatten()


# -------------------------------------- attack node select -------------------------------------- #
# attack node 的选择范围不包含测试集
import heuristic_selection as hs

if args.use_vs_number:
    attach_nodes_size = args.vs_number
else:
    attach_nodes_size = int((data.num_nodes - data.test_mask.sum()) * args.vs_ratio)  # 不包含测试节点个数
print("#Attach Nodes: {}".format(attach_nodes_size))

# 通过 selection_method 从 unlabeled_idx 中筛选中毒节点
idx_attach = None
if args.selection_method == 'none':  # 随机筛选
    idx_attach = hs.obtain_attach_nodes(args, unlabeled_idx, attach_nodes_size)
elif args.selection_method == 'cluster':
    idx_attach = hs.cluster_distance_selection(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx,
                                               not_test_edge_index, attach_nodes_size, device)
elif args.selection_method == 'cluster_degree':
    idx_attach = hs.cluster_degree_selection(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx,
                                             not_test_edge_index, attach_nodes_size, device)
idx_attach = torch.LongTensor(idx_attach).to(device)
print("idx_attach: {}".format(idx_attach))

# unlabeled_idx 要去掉作为触发器节点的部分
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)


# -------------------------------------- trigger generate model -------------------------------------- #
from models.backdoor import Backdoor
from help_funcs import prune_unrelated_edge, prune_unrelated_edge_isolated

model = Backdoor(args, device)
model.fit(features=data.x,
          edge_index=not_test_edge_index,
          edge_weight=None,
          labels=data.y,
          idx_train=idx_train,
          idx_attach=idx_attach,
          idx_unlabeled=unlabeled_idx)
poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()  # 获取到触发器附着后的节点集/边权重/label

# -------------------------------------- defend method -------------------------------------- #
if args.defense_mode == 'prune':
    poison_edge_index, poison_edge_weights = prune_unrelated_edge(args, poison_edge_index, poison_edge_weights,
                                                                  poison_x, device, large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)  # todo: 是否有重复的id？
elif args.defense_mode == 'isolate':
    poison_edge_index, poison_edge_weights, rel_nodes = prune_unrelated_edge_isolated(args, poison_edge_index,
                                                                                      poison_edge_weights, poison_x,
                                                                                      device, large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).tolist()
    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
else:
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)

print("precent of left attach nodes: {:.3f}".format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist())) / len(idx_attach)))


# -------------------------------------- poisoned -------------------------------------- #
from models.construct import model_construct

test_model = model_construct(args, args.test_model, data, device).to(device)
test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,
               train_iters=args.epochs, verbose=False)
output = test_model(poison_x, poison_edge_index, poison_edge_weights)  # 传入触发器到被感染模型进行预测
train_attach_rate = (output.argmax(dim=1)[idx_attach] == args.target_class).float().mean()
print("target class rate on Vs: {:.4f}".format(train_attach_rate))  # 后门攻击的成功率

induct_edge_index = torch.cat([poison_edge_index, test_edge_index], dim=1)
induct_edge_weights = torch.cat(
    [poison_edge_weights, torch.ones([test_edge_index.shape[1]], dtype=torch.float, device=device)])
clean_acc = test_model.test(poison_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)
print("accuracy on clean test nodes: {:.4f}".format(clean_acc))  # clean 测试集上的成功率

