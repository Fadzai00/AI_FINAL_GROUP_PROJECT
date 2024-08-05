# AI_FINAL_GROUP_PROJECT
This project uses Graph Neural Network  (GNN) to predict whether the given SMILES for a drug are for paracetamol or not

YOUTUBE LINK: https://youtu.be/zFnEzmfOALw?si=bre9jIeIitp-3ZKu


Drug Classification with Graph Neural Networks
Overview
This project implements a Graph Neural Network (GNN) for classifying drug compounds based on their SMILES (Simplified Molecular Input Line Entry System) representations. The goal is to distinguish between a target drug (paracetamol) and other compounds.
Requirements
To run this project, you need the following Python libraries:
•	torch (PyTorch)
•	rdkit (for SMILES to graph conversion)
•	scikit-learn (for splitting datasets)
•	pandas (for handling data)
Code Components
Data Preparation
•	Data Loading: Data is loaded from a DataFrame containing SMILES strings and compound names.
•	Graph Conversion: SMILES strings are converted into graph representations suitable for GNN input.
•	Label Assignment: Labels are assigned based on whether the compound is 'paracetamol' or not.
•	Dataset Splitting: The dataset is divided into training and validation sets.
Dataset Class
A custom dataset class is created to handle the graph and label data for training and validation.
Training Function
The training function iterates over the training data, performs forward and backward passes, and updates the model parameters to minimize the loss.
Validation Function
The validation function evaluates the model's performance on the validation set, calculating the accuracy based on predicted and true labels.
Data Loaders
Data loaders are set up for batching the training and validation datasets, enabling efficient model training and evaluation.

Model Definition
GCN Model
A simple Graph Convolutional Network (GCN) model is defined for binary classification. The model consists of two convolutional layers:
•	First Convolutional Layer: Maps from 1 input feature to 64 features.
•	Second Convolutional Layer: Maps from 64 features to 1 output feature, suitable for binary classification.
A global mean pooling layer is used to aggregate node features into a graph-level representation. The final output is processed through a sigmoid activation function to produce probabilities for binary classification.
Initialization
•	Device Setup: The model is initialized to run on a CUDA-enabled GPU if available, otherwise on the CPU.
•	Model: An instance of the GCN class is created.
•	Optimizer: Adam optimizer is used with a learning rate of 0.01.
•	Loss Function: Binary Cross-Entropy Loss (BCELoss) is used for binary classification tasks.
Training and Validation
The training loop runs for a specified number of epochs (50 in this case). During each epoch, the model is trained, and the training loss is printed. Validation accuracy is commented out but can be included for evaluating model performance on the validation set.
Prediction Function
A function predict is defined to make predictions on new SMILES strings:
1.	Graph Conversion: The SMILES string is converted into a graph representation.
2.	Model Evaluation: The model is used to predict the probability of the compound being paracetamol.
3.	Output Interpretation: If the probability is greater than 0.5, the function returns "Paracetamol"; otherwise, it returns "Not Paracetamol".
Testing
An example SMILES string ('CC(=O)Nc1ccc(O)cc1C(=O)O', which corresponds to paracetamol) is used to test the predict function. The result is printed to verify the model's prediction.
UsageEnsure the smiles_to_graph function is correctly implemented to convert SMILES strings into graph representations.
•	Modify hyperparameters, such as learning rate and number of epochs, based on your specific needs and dataset.

Usage
1.	Prepare Data: Ensure your dataset is loaded into a DataFrame with the required columns.
2.	Convert SMILES to Graphs: Implement the necessary function to convert SMILES strings into graph representations.
3.	Train and Validate: Execute the training and validation functions to train the model and evaluate its performance.
4.	Define Model: The GCN class defines the model architecture.
5.	Initialize Components: Set up the model, optimizer, and loss function.
6.	Train Model: Execute the training loop to train the model.
7.	Validate Model: Optionally, validate the model performance on a validation set.
8.	Make Predictions: Use the predict function to classify new SMILES strings.

Notes
•	Ensure that the model, optimizer, and loss function are correctly configured.
•	Adjust hyperparameters such as batch size and learning rate according to your needs.
License
This project is licensed under the MIT License. 

Hosting the Application:
Our application was deployed locally, to host the application use the following command:
streamlit model.py to run the application

