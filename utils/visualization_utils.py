import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_metrics(histories, labels, save_dir):
    """Строит и сохраняет графики метрик обучения для нескольких моделей."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    metrics = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
    for metric in metrics:
        plt.figure()
        for history, label in zip(histories, labels):
            plt.plot(history[metric], label=label)
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"{metric}.png"))
        plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Строит и сохраняет confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8,8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_maps(feature_maps, save_path):
    """Визуализирует feature maps первого conv-слоя."""
    fmap = feature_maps[0]  # первый пример из батча
    n = min(fmap.shape[0], 8)
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    for i in range(n):
        axes[i].imshow(fmap[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_first_layer_activations(activations, save_path):
    """Визуализирует активации первого сверточного слоя."""
    act = activations[0]  # первый пример из батча
    n = min(act.shape[0], 8)  # максимум 8 каналов
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    for i in range(n):
        axes[i].imshow(act[i], cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()