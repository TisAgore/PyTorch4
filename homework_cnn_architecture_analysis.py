import logging
import time
import torch
from models.cnn_models import (ShallowCNN, MediumCNN, DeepCNN, ResidualCNN)
from utils.training_utils import (train, test, get_data_loaders, get_feature_maps)
from utils.comparison_utils import compare_models, count_parameters
from utils.visualization_utils import plot_metrics, plot_feature_maps

logging.basicConfig(filename='Torch/Homework_4/results/architecture_analysis/experiment.log', level=logging.INFO)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, _ = get_data_loaders('CIFAR10', batch_size=128)

    models = [
        ("ShallowCNN", ShallowCNN().to(device)),
        ("MediumCNN", MediumCNN().to(device)),
        ("DeepCNN", DeepCNN().to(device)),
        ("ResidualCNN", ResidualCNN().to(device)),
    ]

    histories = []
    train_times = []
    param_counts = []
    test_accs = []

    for name, model in models:
        print(f"\nОбучение {name}...")
        params = count_parameters(model)
        param_counts.append((name, params))
        print(f"Параметров: {params}")

        start_time = time.time()
        history = train(model, train_loader, test_loader, epochs=10, log_prefix=name,
            grad_flow_plot=True, grad_flow_path=f'Torch/Homework_4/plots/architecture_analysis/grad_{name}.png')
        train_duration = time.time() - start_time
        train_times.append((name, train_duration))
        print(f"Время обучения: {train_duration:.2f} сек")

        _, test_acc = test(model, test_loader)
        test_accs.append((name, test_acc))
        print(f"Test accuracy: {test_acc:.4f}")

        histories.append(history)

        # Визуализация feature maps первого conv-слоя
        images, _ = next(iter(test_loader))
        img = images[0:1].to(device)
        fmap = get_feature_maps(model, img)
        plot_feature_maps(fmap, save_path=f'Torch/Homework_4/plots/architecture_analysis/fmap_{name}.png')

    compare_models(histories, [n for n, _ in models], save_path='Torch/Homework_4/results/architecture_analysis/metrics_depth.json')
    plot_metrics(histories, [n for n, _ in models], save_dir='Torch/Homework_4/plots/architecture_analysis/')

    print("\nСравнение числа параметров:")
    for name, params in param_counts:
        print(f"{name}: {params}")

    print("\nВремя обучения (сек):")
    for name, t in train_times:
        print(f"{name}: {t:.2f}")

    print("\nТочность на тестовом множестве:")
    for name, acc in test_accs:
        print(f"{name}: {acc:.4f}")

if __name__ == '__main__':
    main()
