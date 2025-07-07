import logging
import time
import torch
from models.fc_models import FCNet, DeepFCNet
from models.cnn_models import SimpleCNN, ResNetLike, ResNetCIFAR, ResNetCIFARRegularized
from utils.training_utils import (train, test, get_data_loaders, measure_inference_time, get_predictions)
from utils.comparison_utils import compare_models, count_parameters
from utils.visualization_utils import plot_metrics, plot_confusion_matrix

logging.basicConfig( filename='Torch/Homework_4/results/mnist_comparison/experiment.log', level=logging.INFO)

def main():
    """Сравнивает производительность FC и различных CNN на CIFAR-10 с анализом переобучения, confusion matrix и градиентов."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, class_names = get_data_loaders('CIFAR10', batch_size=128)

    # --- МОДЕЛИ ---
    models = [
        ("DeepFCNet", DeepFCNet().to(device)),
        ("ResNetCIFAR", ResNetCIFAR().to(device)),
        ("ResNetCIFARRegularized", ResNetCIFARRegularized().to(device))
    ]

    histories = []
    train_times = []
    inference_times = []
    param_counts = []
    train_accs = []
    test_accs = []

    for name, model in models:
        logging.info(f"=== {name} ===")
        print(f"\nОбучение {name}...")

        params = count_parameters(model)
        param_counts.append((name, params))
        logging.info(f"Параметров: {params}")
        print(f"Параметров: {params}")

        start_time = time.time()
        history = train(model, train_loader, test_loader, epochs=10, log_prefix=name, grad_flow_plot=True)
        train_duration = time.time() - start_time
        train_times.append((name, train_duration))
        logging.info(f"Время обучения: {train_duration:.2f} сек")
        print(f"Время обучения: {train_duration:.2f} сек")

        train_acc = history['train_acc'][-1]
        train_accs.append((name, train_acc))
        logging.info(f"Train acc: {train_acc:.4f}")
        print(f"Train acc: {train_acc:.4f}")

        test_loss, test_acc = test(model, test_loader)
        test_accs.append((name, test_acc))
        logging.info(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

        inf_time = measure_inference_time(model, test_loader, device)
        inference_times.append((name, inf_time))
        logging.info(f"Время инференса: {inf_time:.4f} сек")
        print(f"Время инференса: {inf_time:.4f} сек")

        histories.append(history)

        # Confusion matrix
        y_true, y_pred = get_predictions(model, test_loader, device)
        plot_confusion_matrix(
            y_true, y_pred, class_names,
            save_path=f'Torch/Homework_4/plots/cifar_comparison/conf_matrix_{name}.png'
        )

    compare_models(histories, [n for n, _ in models], save_path='Torch/Homework_4/results/cifar_comparison/metrics.json')
    plot_metrics(histories, [n for n, _ in models], save_dir='Torch/Homework_4/plots/cifar_comparison/')

    print("\nСравнение числа параметров:")
    for name, params in param_counts:
        print(f"{name}: {params}")

    print("\nВремя обучения (сек):")
    for name, t in train_times:
        print(f"{name}: {t:.2f}")

    print("\nВремя инференса (сек):")
    for name, t in inference_times:
        print(f"{name}: {t:.4f}")

    print("\nТочность на обучающем множестве:")
    for name, acc in train_accs:
        print(f"{name}: {acc:.4f}")

    print("\nТочность на тестовом множестве:")
    for name, acc in test_accs:
        print(f"{name}: {acc:.4f}")

if __name__ == '__main__':
    main()
