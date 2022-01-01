import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, file):
        self.X = 1
        self.Y = 1
        self.centered_X = 1
        self.centered_Y = 1
        self.file = file

    def load_data(self):
        data = loadmat(self.file)
        #print(data)
        x = data['X']
        self.X = x[0,:]
        self.Y = x[1,:]
        return

    def center_data(self):
        meanX = np.sum(self.X)/self.X.shape[0]
        meanY = np.sum(self.Y)/self.Y.shape[0]
        self.centered_X = self.X - meanX
        self.centered_Y = self.Y - meanY
        self.centered_X = np.reshape(self.centered_X, (1,self.X.shape[0]))
        self.centered_Y = np.reshape(self.centered_Y, (1,self.Y.shape[0]))
        return

    def compute_Covariance(self):
        X = np.concatenate((self.centered_X, self.centered_Y), axis = 0)
        sigma = np.dot(X, X.T)
        sigma = (1/self.centered_X.shape[1]) * sigma
        return sigma, X

    def project_data(self, vec, X):
        vec = np.reshape(vec, (vec.shape[0],1))
        proj = np.dot(vec.T, X)
        zer = np.zeros((1, proj.shape[1]))
        plt.scatter(proj, zer)
        plt.show()
        bins = proj.shape[1]
        plt.hist(proj[0,:], bins)
        plt.show()
        return

    def run_algo(self):
        self.load_data()
        self.center_data()
        sigma, data = self.compute_Covariance()
        #print(sigma)
        eigval, eigvec = np.linalg.eig(sigma)
        #print(eigval,eigvec)
        ind = np.argmax(eigval)
        redu = eigvec[:,ind]
        #print("final ",redu)
        self.project_data(redu, data)
        return
