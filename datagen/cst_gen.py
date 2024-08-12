import numpy as np
from scipy.interpolate import splev, splprep
from scipy import optimize
from multiprocessing import Pool
import os
from numpy.linalg import lstsq
from scipy.special import factorial
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class CSTLayer():
    def __init__(self, x_coords=None, n_cst=12, n_x=129, n1=0.5, n2=1.0):
        if x_coords is None:  # Use n_x to generate x_coords
            '''
            Only works for the same x coordinates for both sides of the airfoil
            Airfoil points from upper TE ---> LE ---> lower TE
            n_cst is the order of CST fitting, x_coords is the x coordinates of the original airfoil, and n_x is used when x_coords is not provided
            '''
            self.n_x = n_x
            theta = np.linspace(np.pi, 2*np.pi, n_x)
            self.x_coords = (np.cos(theta) + 1.0)/2
        else:
            self.n_x = len(x_coords)
            self.x_coords = x_coords

        self.n1 = n1
        self.n2 = n2
        self.n_cst = n_cst
        self.A0 = self.A0_matrix()
      
    def A0_matrix(self):
        '''
        y = A0.T.dot(au) + 0.5 * te * x
        '''
        n = self.n_cst
        n1 = self.n1
        n2 = self.n2
        n_x = self.n_x
        x = self.x_coords
        k = np.zeros(n+1)
        A0 = np.zeros([n+1, n_x])
        
        for r in range(n+1):
            k[r] = factorial(n)/factorial(r)/factorial(n-r)
            A0[r, :] = k[r]*x**(n1 + r)*(1 - x)**(n + n2 - r)
        return A0.T
    
    def fit_CST(self, y_coords, n_x=129):
        A0 = self.A0_matrix()
        yu = y_coords[:n_x][::-1]  # Upper surface airfoil, reversing the order from TE to LE to TE
        yl = y_coords[n_x-1:]  # Lower surface airfoil
        te = (yu[-1] - yl[-1])
        au = lstsq(A0, yu - self.x_coords * yu[-1], rcond=None)[0]
        al = lstsq(A0, yl - self.x_coords * yl[-1], rcond=None)[0]
        return au, al, te


def lhs(n, au, al, te, bounds=[-0.02, 0.02]):
    nvars = len(au)  # Number of variables
    result_au = np.zeros((n, nvars))
    result_al = np.zeros((n, nvars))
    result_te = np.full((n, 1), te)
    # Partition each variable
    for i in range(nvars):
        # partitions = np.linspace(au[i] + bounds[0], au[i] + bounds[1], n + 1)
        partitions = np.linspace(au[i] + bounds[0] * (au[i] - al[i]), au[i] + bounds[1] * (au[i] - al[i]), n + 1)
        result_au[:, i] = np.random.permutation(partitions[:-1]) + np.random.uniform(0, abs(bounds[0] - bounds[1]) / n, (n,))
    for i in range(nvars):
        # partitions = np.linspace(al[i] + bounds[0], al[i] + bounds[1], n + 1)
        partitions = np.linspace(al[i] + bounds[0] * (au[i] - al[i]), al[i] + bounds[1] * (au[i] - al[i]), n + 1)
        result_al[:, i] = np.random.permutation(partitions[:-1]) + np.random.uniform(0, abs(bounds[0] - bounds[1]) / n, (n,))
    
    np.random.shuffle(result_au)
    np.random.shuffle(result_al)
    
    return result_au, result_al, result_te


def read_file(file_path):
    data = []
    with open(file_path) as file:
        for line in file:
            values = line.strip().split()
            data.append([float(values[0]), float(values[1])])
    return np.array(data)

def write(file_path, data):
    with open(file_path, 'w') as f:
        for x, y in data:
            f.write(f'{x} {y}\n')


def process_file(file_path):
    data = read_file(file_path)
    y = data[:, 1]
    cst = CSTLayer()
    au, al, te = cst.fit_CST(y)  # x coordinates and number need to be consistent with the original airfoil
    yu = cst.A0.dot(au) + cst.x_coords * te  # cst.x_coords can be replaced with the desired x coordinate distribution
    yl = cst.A0.dot(al) - cst.x_coords * te
    
    n = 10  
    au, al, te = lhs(n, au, al, te)
    for i in range(n):
        yu = cst.A0.dot(au[i]) + cst.x_coords * te[i]
        yl = cst.A0.dot(al[i]) - cst.x_coords * te[i]
        new_y = np.concatenate([yu[::-1], yl[1:]])
        new_data = data.copy()
        new_data[:, 1] = new_y
        file = file_path.split('/')[-1]
        new_path = os.path.join(new_root, file.split('.')[0] + f'_cst_{i}.dat')
        write(new_path, new_data)

# example usage
if __name__ == '__main__':
    dataset_name = 'interpolated_uiuc'
    root_path = f'data/airfoil/{dataset_name}'
    new_root = f'data/airfoil/cst_gen_b'
    os.makedirs(new_root, exist_ok=True)
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    with Pool(processes=8) as pool:
        pool.map(process_file, file_paths)
