import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

class kPCA:
    def __init__(self, file):
        self.X = 1
        self.Y = 1
        self.file = file

    def load_data(self):
        data = loadmat(self.file)
        x = data['X']
        self.X = x[0,:]
        self.Y = x[1,:]
        self.X = np.reshape(self.X, (1,self.X.shape[0]))
        self.Y = np.reshape(self.Y, (1,self.Y.shape[0]))
        return

    def form_Kernel(self):
        z = np.zeros((self.X.shape[1],1))
        Kx = self.X + z
        Ky = self.Y + z
        kernel = ((Kx - self.X.T)**2 + (Ky - self.Y.T)**2)
        kernel = (-1/50) * kernel
        kernel = np.exp(kernel)
        return kernel

    def center_kernel(self, kernel):
        O = np.ones(kernel.shape)
        centered_K = kernel - ((1/kernel.shape[0])*np.dot(O, kernel)) - ((1/kernel.shape[0])*np.dot(kernel, O)) + ((1/(kernel.shape[0]**2))*np.dot(O,np.dot(kernel, O)))
        return centered_K

    def project_data(self, vec, centered_K):
        vec1 = np.reshape(vec[:,0], (vec.shape[0],1))
        proj1 = np.dot(vec1.T, centered_K)
        vec2 = np.reshape(vec[:,1], (vec.shape[0],1))
        proj2 = np.dot(vec2.T, centered_K)
        vec3 = np.reshape(vec[:,2], (vec.shape[0],1))
        proj3 = np.dot(vec3.T, centered_K)
        proj1 = np.real(proj1)
        proj2 = np.real(proj2)
        proj3 = np.real(proj3)
        zer = np.zeros((1, proj1.shape[1]))
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title("Projection of points in Higher Dimensions using Kernel PCA")
        ax.scatter(proj1, proj2, proj3)
        plt.show()

        plt.title("Histogram of Projection of points on 1D using Kernel PCA")
        bins = proj1.shape[1]
        plt.hist(proj1[0], bins)
        plt.show()

        plt.title("Projection of points on 1D using Kernel PCA")
        plt.scatter(proj1, zer)
        plt.show()
        return

    def run_algo(self):
        self.load_data()
        K = self.form_Kernel()
        centered_K = self.center_kernel(K)
        eigval, eigvec = np.linalg.eig(centered_K)
        V = 1/eigval
        V = V**(1/2)
        directions = eigvec * V
        self.project_data(directions, K)
        return
