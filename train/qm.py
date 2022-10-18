import torch
from core.config import cfg, update_cfg
from core.train import run 
from core.model.gnn import GNN
from core.model.setgnn import KCSetGNN
from core.transform import KCSetWLSubgraphs

from torch_geometric.datasets import QM9, QM7b
from torch_geometric.transforms import Constant, Compose
from sklearn.model_selection import train_test_split

# prepare dataset
def create_dataset(cfg):
    torch.set_num_threads(cfg.num_workers)

    # load dataset
    assert cfg.dataset in ['qm9', 'qm7b']
    dataset = QM9(f'data/{cfg.dataset}') if cfg.dataset == 'qm9' else QM7b(f'data/{cfg.dataset}')

    # create transform
    if cfg.subgraph.arch_type == 'KCSetGNN':
        transform = KCSetWLSubgraphs(cfg.subgraph.kmax, cfg.subgraph.stack, cfg.subgraph.kmin, cfg.subgraph.num_components, zero_init=cfg.subgraph.zero_init)

    if dataset.data.x is None:
        transform = Compose([Constant(value=1), transform])
    dataset.transform = transform

    # normalize y 
    y = dataset.data.y
    dataset.data.y = (y-y.mean(dim=0,keepdim=True))/y.std(dim=0, keepdim=True)
    if len(dataset.data.edge_attr.shape) == 1:
        dataset.data.edge_attr = dataset.data.edge_attr.unsqueeze(1)

    # split dataset to train/val/test  #set random_state with 0 to fix split
    train_ratio = 0.8
    idx = torch.arange(len(dataset))
    train_idx, test_idx = train_test_split(idx, test_size=1 - train_ratio, random_state=0)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=0) 

    train_dataset, val_dataset, test_dataset = dataset[train_idx], dataset[val_idx], dataset[test_idx]
    # When without randomness, transform the data to save a bit time
    train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset] 
    test_dataset = [x for x in test_dataset] 
    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    if cfg.model.arch_type == 'GNN':
        model = GNN(1 if cfg.dataset=='qm7b' else 11, 
                    1 if cfg.dataset=='qm7b' else 4, 
                    nhid=cfg.model.hidden_size, 
                    nout=14 if cfg.dataset=='qm7b' else 19, 
                    nlayer=max(cfg.model.num_layers, cfg.model.num_inners), 
                    gnn_type=cfg.model.gnn_type, 
                    dropout=cfg.train.dropout, 
                    res=True)
    if cfg.model.arch_type == 'kWLGNN':
        model = KCSetGNN(1 if cfg.dataset=='qm7b' else 11, 
                         1 if cfg.dataset=='qm7b' else 4,  
                         nhid=cfg.model.hidden_size, 
                         nout=14 if cfg.dataset=='qm7b' else 19, 
                         nlayer_intra=cfg.model.num_inners, 
                         nlayer_inter=cfg.model.num_layers,
                         gnn_type=cfg.model.gnn_type,
                         bgnn_type=cfg.model.bgnn_type, 
                         dropout=cfg.train.dropout, 
                         res=True, 
                         pools=cfg.model.pools,
                         mlp_layers=2,
                         num_bipartites=cfg.subgraph.kmax-1-cfg.subgraph.kmin if cfg.subgraph.stack is True else 1,
                         half_step=cfg.model.half_step) # num_bipartites should remove min, now min=0
    return model

def train(train_loader, model, optimizer, device, scaler):
    total_loss = 0
    N = 0 
    for data in train_loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            loss = (model(data) - y).abs().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * num_graphs
        N += num_graphs
    return total_loss / N

@torch.no_grad()
def test(loader, model, evaluator, device):
    total_error = 0
    N = 0
    for data in loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        total_error += (model(data) - y).abs().mean().item() * num_graphs
        N += num_graphs
    test_perf = - total_error / N
    return test_perf
    

if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/config/qm.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test, use_amp=cfg.amp)

