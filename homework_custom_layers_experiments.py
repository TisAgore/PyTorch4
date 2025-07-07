import logging
import time
import torch
from models.cnn_models import BasicResNet, BottleneckResNet, WideResNet
from utils.training_utils import train, test, get_data_loaders
from utils.comparison_utils import compare_models, count_parameters
from utils.visualization_utils import plot_metrics

logging.basicConfig(filename='Torch/Homework_4/results/custom_layers/experiment.log', level=logging.INFO)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, _ = get_data_loaders('CIFAR10', batch_size=128)

    models = [
        ("BasicResNet", BasicResNet().to(device)),
        ("BottleneckResNet", BottleneckResNet().to(device)),
        ("WideResNet", WideResNet().to(device)),
    ]

    histories = []
    param_counts = []
    train_times = []
    test_accs = []

    for name, model in models:
        print(f"\nОбучение {name}...")
        params = count_parameters(model)
        param_counts.append((name, params))
        print(f"Параметров: {params}")

        start_time = time.time()
        history = train(model, train_loader, test_loader, epochs=20, log_prefix=name)
        train_duration = time.time() - start_time
        train_times.append((name, train_duration))
        print(f"Время обучения: {train_duration:.2f} сек")

        _, test_acc = test(model, test_loader)
        test_accs.append((name, test_acc))
        print(f"Test accuracy: {test_acc:.4f}")

        histories.append(history)

    compare_models(histories, [n for n, _ in models], save_path='Torch/Homework_4/results/custom_layers/metrics_resblocks.json')
    plot_metrics(histories, [n for n, _ in models], save_dir='Torch/Homework_4/plots/custom_layers/')

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