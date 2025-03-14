#pseudo-likelihood training of the EA 2d model using data at a given temperature
#we perform the inference of the J parameters for differnet values of P
#we then perform zero temperature dynamics using the inferred J
#the dynamics is carried out both for the training examples and for other P random configurations
#the zero-temperarture dynamics checks whetere the patterns are stable: we check the final Mattis magnetization of the data

#standard imports
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#imports of custom files
sys.path.append("../../Legacy/packages")
from geometry import *
from utilities import *
from global_steps import *
from data_loads import *
from tqdm import tqdm
import argparse

class PLLModel(nn.Module):
    """PLL model."""
    def __init__(self, input_dim, output_dim):
        super(PLLModel, self).__init__()
        # Linear layer without bias
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):
        # Apply the linear layer
        x = self.linear(x)
        # Apply the sigmoid activation
        x = torch.sigmoid(2*x)
        return x

def train_pll_model(model, X_train, y_train, learning_rate=0.001, batch_size=32, epochs=50, decay_factor = 0.999):
    """
    Train the PLLModel using Adam optimizer with manual minibatches, data already on GPU.

    Args:
        model: The PLLModel instance to be trained.
        X_train (torch.Tensor): Training input data on the GPU.
        y_train (torch.Tensor): Training labels on the GPU.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        batch_size (int): Size of the minibatches for training. Default is 32.
        epochs (int): Number of training epochs. Default is 50.

    Returns:
        model: The trained model.
    """
    # Ensure data is on the same device as the model
    device = next(model.parameters()).device
    X_train, y_train = X_train.to(device), y_train.to(device)
    
    # Define loss function (Binary Cross-Entropy Loss for a sigmoid output)
    criterion = nn.BCELoss(reduction='mean')
    
    # Define optimizer (Adam)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Determine the number of samples
    num_samples = X_train.size(0)
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle indices for random minibatches
        indices = torch.randperm(num_samples, device=device)
        running_loss = 0.0
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract minibatch
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            
            # Compute loss
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()

            # Exponentially decay the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * (decay_factor ** epoch)

            # Zero the diagonal elements of the weight matrix
            with torch.no_grad():
                weight_matrix = model.linear.weight.data  # Access the weight matrix data directly
                diag_indices = torch.arange(weight_matrix.size(0), device=weight_matrix.device)
                weight_matrix[diag_indices, diag_indices] = 0.0  # Zero the diagonal elements of the weight matrix  
            # Accumulate loss for reporting
            running_loss += loss.item()
        torch.cuda.empty_cache()
    return model


def zero_temperature_dynamics(W, x):
    with torch.no_grad():
        """Run the zero temperature dynamics for the model."""
        y = x.clone()
        y = torch.einsum("ij, kj->ki", W, y)
        return torch.sign(y)


if __name__ == "__main__":
    # Example usage or main function code
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PLLModel with specified temperature.')
    parser.add_argument('--temperature', type=str, required=True, help='Temperature value for the simulation')
    args = parser.parse_args()

    # Use the temperature argument
    T = args.temperature

    #load the data. You can choose the size and the seed here
    #seed = 3397150145
    #L = 16
    seed = 3034024510
    L = 32
    N = L*L
    data, J, _, _ = get_data(L, T, seed, RUN = 1, back = "../../../Dati/Alpha", ordering = "raster")
    data2, _, _, _ = get_data(L, T, seed, RUN = 1, back = "../../../Dati/Alpha", ordering = "raster")
    
    #and set the simulations parameters
    zero_temp_dynamics_steps = 100 #number of zero temperature dynamics steps to perform (both for training and test)
    Pvalues = np.concatenate((np.arange(1, 30, 2), np.logspace(np.log10(31), np.log10(65536), num=200, dtype=int))) # number of training data on which to loop

    T = float(T)

    all_ms = []
    for P in Pvalues:
        indices = torch.randperm(data.size(0))[:P]
        data_to_use = data[indices].clone()

        model = PLLModel(input_dim=N, output_dim=N)
        model = model.to("cuda")
        model = train_pll_model(model, data_to_use, (data_to_use+1)/2, learning_rate=100, batch_size=P, epochs=1000, decay_factor=0.999)

        weight_matrix = model.linear.weight.detach()
        Jinf = (weight_matrix + weight_matrix.T) / 2

        frac = 0  # Fraction of spins to flip
        starting_point = data_to_use.clone()
        num_flips = int(frac * starting_point.numel())
        flip_indices = torch.randperm(starting_point.numel(), device=DEVICE)[:num_flips]
        starting_point.view(-1)[flip_indices] *= -1
        
        result = data_to_use.clone()
        with torch.no_grad():
            for i in range(zero_temp_dynamics_steps):
                result = zero_temperature_dynamics(model.linear.weight, result)
        magnetizations = ((N-torch.sum(torch.abs(result-data_to_use)/2, axis = 1))/N).mean()
        energy_training = compute_energy(result, Jinf, take_mean=True)/N

        #test set
        indices = torch.randperm(data.size(0))[:P]
        data_test = data2[indices].clone()
        result = data_test.clone()
        with torch.no_grad():
            for i in range(zero_temp_dynamics_steps):
                result = zero_temperature_dynamics(model.linear.weight, result)
        magnetizations_test = ((N-torch.sum(torch.abs(result-data_test)/2, axis = 1))/N).mean()
        energy_test = compute_energy(result, Jinf, take_mean=True)/N

        #random
        result = (torch.randint(0, 2, (P, N), device=DEVICE).float() * 2.) - 1.
        start_random = result.clone()
        with torch.no_grad():
            for i in range(zero_temp_dynamics_steps):
                result = zero_temperature_dynamics(model.linear.weight, result)
        magnetizations_random = ((N-torch.sum(torch.abs(result-start_random)/2, axis = 1))/N).mean()
        energy_random = compute_energy(result, Jinf, take_mean=True)/N

        #GS
        GS = torch.ones((2, N), device = "cuda")
        result = torch.ones((2, N), device = "cuda")
        with torch.no_grad():
            for i in range(zero_temp_dynamics_steps):
                result = zero_temperature_dynamics(model.linear.weight, result)
        magnetizations_GS = ((N-torch.sum(torch.abs(result-GS)/2, axis = 1))/N).mean()
        energy_GS = compute_energy(result, Jinf, take_mean=True)/N

        #weight_matrix_end = weight_matrix.T
        tgt = J/T
        gamma = torch.sqrt(((Jinf-tgt)**2).sum()/(tgt**2).sum())
        R = torch.sum(Jinf*tgt)/torch.norm(tgt)/torch.norm(Jinf)

        all_ms.append([P, magnetizations, magnetizations_test, magnetizations_GS, magnetizations_random, energy_training, energy_test, energy_GS, energy_random, gamma, R])
        
        #comment out to avoid print 
        #print(P, float(magnetizations), float(magnetizations_test), float(magnetizations_GS), float(energy_training), float(energy_test), float(energy_GS), float(gamma), float(R))
    all_ms = np.array(all_ms)
    
    with open(f'../../../Dati/Omega/Results/L{L}_seed{seed}/L{L}_seed{seed}_T{T:.2f}.txt', 'a') as f:
        np.savetxt(f, all_ms)