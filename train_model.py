import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn, optim
from models.attention_Unet import AttentionUNet
from data.setup_dataset import TumorDataset
from utils.eval_test_model import evaluate_model, test_model
import os

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TumorDataset(data_dir="data/train")
    val_dataset = TumorDataset(data_dir="data/val")
    test_dataset = TumorDataset(data_dir="data/test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = AttentionUNet(in_ch=1, out_ch=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        loop = tqdm((train_loader), total = len(train_loader))
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.detach().item()
            train_loss = running_loss / len(train_loader)
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            loop.set_postfix(train_loss = train_loss)
        
        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_dice, val_precision, val_recall = evaluate_model(model, val_loader, criterion, device)
        print(f"{'Validation':<12}: Val Loss = {val_loss:.4f}, Dice = {val_dice:.4f}, Precision = {val_precision:.4f}, Recall = {val_recall:.4f}")


        if (epoch + 1) % 10 == 0:
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/attention_unet_epoch{epoch+1}.pth")

    test_model(model, test_loader, device, output_dir="results")

if __name__ == "__main__":
    train_model()
