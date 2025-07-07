import json
import torch

def compare_models(histories, labels, save_path):
    """Сохраняет сравнение историй обучения разных моделей в файл."""
    results = {}
    for label, history in zip(labels, histories):
        results[label] = {
            'train_loss': history['train_loss'],
            'test_loss': history['test_loss'],
            'test_acc': history['test_acc'],
        }
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

def count_parameters(model):
    """Возвращает количество обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
