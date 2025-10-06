import torch
from torch import nn
import numpy as np
from functools import partial

# Import from segregated files
from model import NeuralNet
from optimizer import run_smo, objective_function
from utils import load_and_preprocess_data, accuracy_fn, plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import classification_report

def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device.")

    # Data loading and preprocessing
    # IMPORTANT: Update these paths to your actual file locations
    file_paths = [
        './data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
        './data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
        './data/Friday-WorkingHours-Morning.pcap_ISCX.csv',
        './data/Monday-WorkingHours.pcap_ISCX.csv',
        './data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
        './data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
        './data/Tuesday-WorkingHours.pcap_ISCX.csv',
        './data/Wednesday-workingHours.pcap_ISCX.csv',
    ]

    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(file_paths)
    except FileNotFoundError:
        print("Error: One or more data files not found.")
        print("Please create a 'data' directory and place the CIC-IDS-2017 CSV files in it.")
        print("You can download the dataset from: https://www.unb.ca/cic/datasets/ids-2017.html")
        return

    # Convert data to PyTorch Tensors
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)

    # Initialize model and loss function
    input_features = X_train.shape[1]
    model = NeuralNet(input_features=input_features).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizer parameters
    # Note: These parameters are computationally intensive.
    # Reduce PopSize and iters for a faster, less thorough run.
    dim = 2496 + 32 + 1024 + 32 + 32 + 1 # Total number of weights and biases
    lb = -1
    ub = 1
    pop_size = 20   # Population size (e.g., 50)
    iters = 50      # Number of iterations (e.g., 100)
    acc_err = 1e-10

    # Create a partial function for the objective function with fixed arguments
    objf_partial = partial(
        objective_function,
        model=model,
        loss_fn=loss_fn,
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        device=device
    )

    # Run Spider Monkey Optimization
    print("Starting Spider Monkey Optimization...")
    best_weights = run_smo(
        objf=objf_partial,
        lb=np.full(dim, lb),
        ub=np.full(dim, ub),
        dim=dim,
        pop_size=pop_size,
        iters=iters,
        acc_err=acc_err
    )
    print("Optimization finished.")

    # Set final model weights from optimizer result
    objective_function(best_weights, model, loss_fn, X_train_tensor, y_train_tensor, device)

    # Evaluation
    print("\n--- Final Model Evaluation ---")
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor).squeeze()
        test_pred_probs = torch.sigmoid(test_logits)
        test_pred_labels = torch.round(test_pred_probs)

    # Convert to CPU for scikit-learn functions
    y_test_cpu = y_test_tensor.cpu().numpy()
    test_pred_labels_cpu = test_pred_labels.cpu().numpy()
    test_pred_probs_cpu = test_pred_probs.cpu().numpy()

    # Print results and plots
    print("\nClassification Report:")
    print(classification_report(y_test_cpu, test_pred_labels_cpu, target_names=['Benign', 'Attack']))
    
    plot_confusion_matrix(y_test_cpu, test_pred_labels_cpu)
    plot_roc_curve(y_test_cpu, test_pred_probs_cpu)

if __name__ == '__main__':
    main()
