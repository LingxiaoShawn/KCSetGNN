import torch
from core.config import cfg, update_cfg
from core.train import run 
from core.model.setgnn import KCSetGNN
from torch_geometric.datasets import ZINC
from core.transform import KCSetWLSubgraphs


def create_dataset(cfg): 
    torch.set_num_threads(cfg.num_workers)
    transform = transform_eval = KCSetWLSubgraphs(cfg.subgraph.kmax, cfg.subgraph.stack, cfg.subgraph.kmin, cfg.subgraph.num_components, zero_init=cfg.subgraph.zero_init)
    root = 'data/ZINC'
    train_dataset = ZINC(root, subset=True, split='train', transform=transform)
    val_dataset = ZINC(root, subset=True, split='val', transform=transform_eval) 
    test_dataset = ZINC(root, subset=True, split='test', transform=transform_eval)

    train_dataset = [x for x in train_dataset] 
    val_dataset = [x for x in val_dataset] 
    test_dataset = [x for x in test_dataset]   
    return train_dataset, val_dataset, test_dataset


def create_model(cfg):
    model = KCSetGNN(None, None, 
                    nhid=cfg.model.hidden_size, 
                    nout=1, 
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
            loss = (model(data).squeeze() - y).abs().mean()
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
        total_error += (model(data).squeeze() - y).abs().mean().item() * num_graphs
        N += num_graphs
    test_perf = - total_error / N
    return test_perf

if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/config/zinc.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test, use_amp=cfg.amp)  