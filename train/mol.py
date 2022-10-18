import torch
from core.config import cfg, update_cfg
from core.train import run 
from core.model.setgnn import KCSetGNN
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from core.transform import KCSetWLSubgraphs

def create_dataset(cfg): 
    # No need to do offline transformation
    torch.set_num_threads(cfg.num_workers)
    transform = transform_eval = KCSetWLSubgraphs(cfg.subgraph.kmax, 
                                                  cfg.subgraph.stack, 
                                                  cfg.subgraph.kmin, 
                                                  cfg.subgraph.num_components, 
                                                  zero_init=cfg.subgraph.zero_init)

    dataset = PygGraphPropPredDataset(cfg.dataset, 'data')
    cfg.num_tasks = dataset.num_tasks
    split_idx = dataset.get_idx_split()
    train_dataset, val_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_dataset.transform, val_dataset.transform, test_dataset.transform = transform, transform_eval, transform_eval

    # When without randomness, transform the data to save a bit time
    train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset] 
    test_dataset = [x for x in test_dataset] 

    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    model = KCSetGNN(None, None, 
                     nhid=cfg.model.hidden_size, 
                     nout=cfg.num_tasks, 
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


if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/config/mol.yaml')
    cfg = update_cfg(cfg)
    # cfg.device = 'cpu'
    evaluator = Evaluator(cfg.dataset)

    if cfg.dataset in ['ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo']:
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
                    loss = (model(data).squeeze() - y.squeeze()).abs().mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * num_graphs
                N += num_graphs
            return total_loss / N

        @torch.no_grad()
        def test(loader, model, evaluator, device):
            y_preds, y_trues = [], []
            for data in loader:
                data = data.to(device)
                y_preds.append(model(data).squeeze())
                y_trues.append(data.y.squeeze())
            y_preds = torch.cat(y_preds, dim=0)
            y_trues = torch.cat(y_trues, dim=0)
            # (y_preds - y_trues).mean()
            return -(y_preds - y_trues).abs().mean()
    else:

        def train(train_loader, model, optimizer, device, scaler):
            total_loss = 0
            N = 0
            criterion = torch.nn.BCEWithLogitsLoss()
            for data in train_loader:
                data = data.to(device)
                mask = ~torch.isnan(data.y)
                y = data.y.to(torch.float)[mask]

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    out = model(data)[mask]
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * data.num_graphs
                N += data.num_graphs
            return total_loss / N

        @torch.no_grad()
        def test(loader, model, evaluator, device):
            y_preds, y_trues = [], []
            for data in loader:
                data = data.to(device)
                y_preds.append(model(data))
                y_trues.append(data.y)

            return evaluator.eval({
                'y_pred': torch.cat(y_preds, dim=0),
                'y_true': torch.cat(y_trues, dim=0),
            })[evaluator.eval_metric]

    run(cfg, create_dataset, create_model, train, test, evaluator=evaluator, use_amp=cfg.amp)   