import torch
from core.config import cfg, update_cfg
from core.train import run 
from core.model.setgnn import KCSetGNN
from core.model.ppgn import PPGN
from core.transform import KCSetWLSubgraphs
from core.data import CFI

def create_dataset(cfg): 
    # No need to do offline transformation
    torch.set_num_threads(cfg.num_workers)
    transform = KCSetWLSubgraphs(cfg.subgraph.kmax, cfg.subgraph.stack, cfg.subgraph.kmin, cfg.subgraph.num_components, zero_init=cfg.subgraph.zero_init) if cfg.model.arch_type == 'KCSetGNN' else None
    root = 'data'
    dataset = CFI(root, transform=transform, grohe=True)

    i = cfg.task*2
    dataset_list = [x for x in dataset[i:i+2]] 

    # When without randomness, transform the data to save a bit time
    train_dataset = dataset_list
    val_dataset = dataset_list
    test_dataset = dataset_list
    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    if cfg.model.arch_type == 'PPGN':
        model = PPGN(None, None,
                     nhid=cfg.model.hidden_size, 
                     nout=2, 
                     nlayer=cfg.model.num_layers)

    else:
        model = KCSetGNN(None, None, 
                        nhid=cfg.model.hidden_size, 
                        nout=2, 
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

    # print(model)
    return model

def train(train_loader, model, optimizer, device, scaler):
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = model(data).squeeze()
            loss = criterion(out, data.y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader, model, evaluator, device):
    model.train()
    y_preds, y_trues = [], []
    for data in loader:
        data = data.to(device)
        y_preds.append(torch.argmax(model(data), dim=-1))
        y_trues.append(data.y)
    y_preds = torch.cat(y_preds, -1)
    y_trues = torch.cat(y_trues, -1)
    return (y_preds == y_trues).float().mean()

if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/config/cfi.yaml')
    cfg = update_cfg(cfg)
    # cfg.device = 'cpu'
    run(cfg, create_dataset, create_model, train, test)