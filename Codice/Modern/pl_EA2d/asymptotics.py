#similarly to the pseudolikelihood program, we compute the asymptotics
#we consider the case of the largest P we have, and what happens when using J

#standard imports
from pseudolikelihood_EA2d import *

if __name__ == "__main__":
    # Example usage or main function code
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PLLModel with specified temperature.')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature value for the simulation')
    args = parser.parse_args()

    # Use the temperature argument
    T = args.temperature

    #load the data
    seed = 3397150145
    L = 16
    N = L*L
    data, J, _, _ = get_data(L, T, seed, RUN = 1, back = "../../../Dati/Alpha", ordering = "raster")
    data2, _, _, _ = get_data(L, T, seed, RUN = 2, back = "../../../Dati/Alpha", ordering = "raster")
    
    #and set the simulations parameters
    zero_temp_dynamics_steps = 1000 #number of zero temperature dynamics steps to perform (both for training and test)
    P = len(data)

    #clone the data for usage
    data_to_use = data.clone()

    #train the model
    model = PLLModel(input_dim=N, output_dim=N)
    model = model.to("cuda")
    model = train_pll_model(model, data_to_use, (data_to_use+1)/2, learning_rate=1, batch_size=P, epochs=5000)
    
    #perform 1000 steps of zero temperature dynamics
    result = data_to_use.clone()
    with torch.no_grad():
        for i in range(zero_temp_dynamics_steps):
            result = zero_temperature_dynamics(model.linear.weight.T, result)
    
    #Compute the Mattis magnetizations with respsect to the training data
    magnetizations = ((N-torch.sum(torch.abs(result-data_to_use)/2, axis = 1))/N).mean()
    energy_training = compute_energy(result, J, take_mean=True)
    segni = torch.sign(result.sum(axis = 1))
    closer = torch.einsum("ij, i->ij", result, segni)
    gs_distance_training = torch.abs(closer-1).mean()/2

    #test set
    data_test = data2.clone()
    result = data_test.clone()
    #perform the zero temperature dyamics for the test set
    with torch.no_grad():
        for i in range(zero_temp_dynamics_steps):
            result = zero_temperature_dynamics(model.linear.weight.T, result)

    #Compute the Mattis magnetizations with respect to the test data
    magnetizations_test = ((N-torch.sum(torch.abs(result-data_test)/2, axis = 1))/N).mean()
    energy_test = compute_energy(result, J, take_mean=True)
    segni = torch.sign(result.sum(axis = 1))
    closer = torch.einsum("ij, i->ij", result, segni)
    gs_distance_test = torch.abs(closer-1).mean()/2

    #perfect case
    data_test = data2.clone()
    result = data_test.clone()
    with torch.no_grad():
        for i in range(zero_temp_dynamics_steps):
            result = zero_temperature_dynamics(J, result)

    #Compute the Mattis magnetizations with respect to the test data
    magnetizations_perfect = ((N-torch.sum(torch.abs(result-data_test)/2, axis = 1))/N).mean()
    energy_perfect = compute_energy(result, J, take_mean=True)
    segni = torch.sign(result.sum(axis = 1))
    closer = torch.einsum("ij, i->ij", result, segni)
    gs_distance_perfect = torch.abs(closer-1).mean()/2

    #save the results
    torch.cuda.empty_cache()
    with open(f'../../../Dati/Omega/Results/L{L}_seed{seed}/asymptotics.txt', 'a') as f:
        f.write(f"{T} {magnetizations.item()} {magnetizations_test.item()} {magnetizations_perfect.item()} {energy_training.item()} {energy_test.item()} {energy_perfect.item()} {gs_distance_training.item()} {gs_distance_test.item()} {gs_distance_perfect.item()}\n")