CIC-IDS-2017 Intrusion Detection with a Neural Network and Spider Monkey Optimizer
This project demonstrates training a simple neural network for network intrusion detection on the CIC-IDS-2017 dataset, using a custom Spider Monkey Optimization (SMO) algorithm instead of traditional gradient-based methods like SGD or Adam.

File Structure
main.py: The main script to execute the entire data processing, training, and evaluation pipeline.

model_architecture.py: Contains the PyTorch neural network class (NeuralNet).

smo_optimizer.py: Implements the Spider Monkey Optimization (SMO) algorithm.

preprocessing_and_metric_functions.py: Includes helper functions for data loading, preprocessing, and plotting evaluation metrics (confusion matrix, ROC curve).

README.md: This file.

Setup
Prerequisites:

Python 3.8+

PyTorch

scikit-learn

pandas

NumPy

Matplotlib

Seaborn

Install Dependencies:

pip install torch pandas scikit-learn numpy matplotlib seaborn

Dataset:

Download the CIC-IDS-2017 dataset CSV files from the official source.

Create a directory named data in the root of the project.

Place all the downloaded .csv files inside the data directory.

How to Run
Ensure you have completed the setup steps, including downloading the dataset and placing it in the data/ folder.

Execute the main script from your terminal:

python main.py

Note: The Spider Monkey Optimization algorithm is computationally expensive. The default parameters in main.py (pop_size=20, iters=50) are set for a relatively quick demonstration. For potentially better results, you can increase these values, but be prepared for significantly longer training times.