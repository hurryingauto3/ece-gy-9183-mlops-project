# utils.py
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

def collate_fn(batch):
    x_list, y_list, fips_list, crop_list = zip(*batch)
    x_pad = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True)
    y = torch.stack(y_list)
    fips = torch.tensor(fips_list)
    crops = torch.tensor(crop_list)
    return x_pad, y, fips, crops

def train_model(model, train_loader: DataLoader,
                num_epochs: int = 20,
                lr: float       = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    for ep in range(num_epochs):
        model.train()
        total_loss = 0
        for x,y,f in tqdm.tqdm(train_loader, desc=f"Epoch {ep+1}"):
            x,y,f = x.to(device), y.to(device), f.to(device)
            opt.zero_grad()
            pred = model(x,f)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss/len(train_loader)
        print(f" → epoch {ep+1} avg train loss: {avg:.4f}")

    return model

def evaluate_model(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ys_true, ys_pred = [], []
    with torch.no_grad():
        for x,y,f in loader:
            x,y,f = x.to(device), y.to(device), f.to(device)
            out = model(x,f)
            ys_true.extend(y.cpu().numpy())
            ys_pred.extend(out.cpu().numpy())
    rmse = mean_squared_error(ys_true, ys_pred, squared=False)
    mae  = mean_absolute_error(ys_true, ys_pred)
    print(f"→ RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae
