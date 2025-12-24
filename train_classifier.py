import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from dataloader import DiffusionDataset
from model import MNISTClassifier

# Assuming MNISTClassifier and ResBlock are imported or defined above
# from model import MNISTClassifier 

def train_classifier(args):
    os.makedirs(args.weights_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Setup Data (Reusing your DiffusionDataset structure)
    train_dataset = DiffusionDataset(file_path=args.train_file_path, mode="train", steps=args.steps)
    val_dataset = DiffusionDataset(file_path=args.val_file_path, mode="test", steps=args.steps)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Model, Loss, Optimizer
    model = MNISTClassifier(image_channels=args.img_ch, time_dim=args.time_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # noisy_image (x_t), actual_noise (unused), time (t), label (y)
            x_t, _, t, y = batch
            
            x_t, t, y = x_t.to(device), t.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_t, t)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        if epoch % 5 == 0 or epoch == args.epochs-1 : 
            # 3. Validation Loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for vbatch in val_loader:
                    x_t_v, _, t_v, y_v = vbatch
                    x_t_v, t_v, y_v = x_t_v.to(device), t_v.to(device), y_v.to(device)

                    logits_v = model(x_t_v, t_v)
                    v_loss = criterion(logits_v, y_v)
                    val_loss += v_loss.item()
                    
                    # Track Accuracy
                    preds = logits_v.argmax(dim=1)
                    correct += (preds == y_v).sum().item()
                    total += y_v.size(0)

            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.2f}%")

            # 4. Save Best and Last
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{args.weights_dir}/classifier_best.pt")
                print("--- Saved Best Model ---")
            
        torch.save(model.state_dict(), f"{args.weights_dir}/classifier_last.pt")

if __name__ == "__main__":
    # Minimal Args for the Classifier
    class Args:
        train_file_path = "mnist_data.pkl"
        val_file_path = "mnist_data.pkl"
        weights_dir = "classifier_weights"
        steps = 1000
        img_ch = 1
        time_dim = 128
        learning_rate = 1e-4
        epochs = 100
        batch_size = 64
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_classifier(Args())