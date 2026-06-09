import torch
import os

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train(model, train_loader, val_loader,
          epochs=15, lr=0.0005, device='cpu', save_name='model'):

    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc  = correct / total
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = os.path.join(CHECKPOINT_DIR, f'{save_name}_best.pth')
            torch.save(model.state_dict(), path)
            print(f"  ✓ Checkpoint saved ({val_acc:.4f})")

    return history
