import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time

def get_data_loaders(dataset_name, batch_size=128):
    """Загружает и возвращает загрузчики данных для указанного датасета."""
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST('data', train=True, download=True, transform=transform)
        test = datasets.MNIST('data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
        class_names = [str(i) for i in range(10)]
        return train_loader, test_loader, class_names
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        train = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        test = datasets.CIFAR10('data', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2)
        class_names = train.classes
        return train_loader, test_loader, class_names
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_name}")

def train(model, train_loader, test_loader, epochs=20, log_prefix="", grad_flow_plot=False, grad_flow_path=None):
    device = next(model.parameters()).device
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            if grad_flow_plot and grad_flow_path and epoch % 5 == 0:
                plot_grad_flow(model.named_parameters(), grad_flow_path.replace('.png', f'_epoch{epoch}.png'))
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        test_loss, test_acc = test(model, test_loader)
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        print(f"{log_prefix} Epoch {epoch+1}: train_loss={avg_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    return history

def test(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    return loss, acc

def plot_grad_flow(named_parameters, save_path):
    """Визуализирует средние значения градиентов по слоям."""
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    plt.figure(figsize=(8,4))
    plt.plot(ave_grads, alpha=0.7, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=8)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Слои")
    plt.ylabel("Средний градиент")
    plt.title("Градиентный поток")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_feature_maps(model, img_tensor):
    """Возвращает feature maps первого conv-слоя для одного изображения."""
    model.eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                return module(img_tensor).cpu().numpy()
    return None

def measure_inference_time(model, data_loader, device):
    """Измеряет среднее время инференса на одном батче."""
    model.eval()
    times = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            start = time.time()
            output = model(data)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            times.append(end - start)
            break  # только один батч для оценки
    return sum(times) / len(times) if times else 0

def get_predictions(model, data_loader, device):
    """Возвращает истинные и предсказанные метки для всего датасета."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(target.numpy())
    return np.array(y_true), np.array(y_pred)

def get_first_layer_activations(model, img_tensor):
    """Возвращает активации первого сверточного слоя для одного изображения."""
    model.eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                return module(img_tensor).cpu().numpy()
    return None