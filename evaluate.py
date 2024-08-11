import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils import Fit_airfoil

#Calculate smoothness
def calculate_smoothness(airfoil):
    smoothness = 0.0
    num_points = airfoil.shape[0]
    for i in range(num_points):
        p_idx = (i - 1) % num_points
        q_idx = (i + 1) % num_points

        p = airfoil[p_idx]
        q = airfoil[q_idx]

        if p[0] == q[0]:
            distance = abs(airfoil[i, 0] - p[0])
        else:
            m = (q[1] - p[1]) / (q[0] - p[0])
            b = p[1] - m * p[0]
            distance = abs(m * airfoil[i, 0] - airfoil[i, 1] + b) / np.sqrt(m**2 + 1)
        smoothness += distance
    return smoothness

# Calculate PARSEC loss and smoothness
def calculate_parsec_loss(source, target):
    source_parsec = Fit_airfoil(source).parsec_features
    target_parsec = Fit_airfoil(target).parsec_features
    parsec_loss = [[0]*3 for _ in range(11)]
    for i2 in range(11):
        parsec_loss[i2][0] += abs(source_parsec[i2] - target_parsec[i2])  # Absolute error
        parsec_loss[i2][1] += abs(source_parsec[i2] - target_parsec[i2]) / (abs(target_parsec[i2]) + 1e-9)  # Relative error
        parsec_loss[i2][2] += abs(target_parsec[i2])  # Absolute value of the true value

    return parsec_loss

# Calculate diversity score
def cal_diversity_score(data, subset_size=10, sample_times=1000):
    # Average log determinant
    N = data.shape[0]
    data = data.reshape(N, -1) 
    mean_logdet = 0
    for i in range(sample_times):
        ind = np.random.choice(N, size=subset_size, replace=False)
        subset = data[ind]
        D = squareform(pdist(subset, 'euclidean'))
        S = np.exp(-0.5*np.square(D))
        (sign, logdet) = np.linalg.slogdet(S)
        mean_logdet += logdet
    return mean_logdet/sample_times


# Example usage
if __name__ == "__main__":
    # Example airfoil data with 257 points (each point has x and y coordinates)
    airfoil_example = np.random.rand(257, 2)

    # Calculate smoothness
    smoothness_value = calculate_smoothness(airfoil_example)
    print(f"Smoothness: {smoothness_value}\n")

    # Example source and target airfoil data
    source_airfoil = np.random.rand(257, 2) 
    target_airfoil = np.random.rand(257, 2) 

    # Calculate PARSEC loss
    parsec_loss_value = calculate_parsec_loss(source_airfoil, target_airfoil)
    print(f"PARSEC Loss: {parsec_loss_value}\n")

    # Example data for diversity score calculation
    airfoil_set = np.random.rand(20, 257, 1)  # A set of 20 random airfoils, each with y [257, 1] (same x)

    # Calculate diversity score
    diversity_score = cal_diversity_score(airfoil_set)
    print(f"Diversity Score: {diversity_score}")
