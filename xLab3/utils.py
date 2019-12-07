from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.patches import Ellipse

def generate_data(n_clusters=None, n_dim=None, n_data=None, mus=None, sigmas=None):
    """
    Generates random data with respect to the parameters provided
    Args:
        n_clusters (int): the number of clusters
        n_dim (int): data dimensionality
        n_data (int, list): total number of data (split evenly into clusters)
                            if list, number of datapoints per cluster
        mus (list): list of  means, can be ints of lists, depending on
                    dimensionality. If int for multivariate data, all
                    dimensions have same mean.
        sigmas (list): A list of variances, can be int, or matrix
    Returns:
        X (np.array): (n_data, n_dim) or (sum(n_data), n_dim)
    """
    if n_clusters is None:
        n_clusters = len(mus)

    if mus is None:
        mus = np.random.uniform(-5, 5, size=(n_clusters,n_dim)).tolist()

    if sigmas is None:
        sigmas_ = np.random.uniform(-.7, .7, size=(n_clusters,n_dim))
        sigmas = []
        for k in range(n_clusters):
            sigmas.append(sigmas_[k,:] @ sigmas_[k,:].T)

    X_ = []
    if isinstance(n_data, int):
        n_data = [int(n_data/n_clusters) for _ in range(n_clusters)]

    for mu, sigma, n in zip(mus, sigmas, n_data):
        if np.shape(sigma) == (n_dim, n_dim):
            if not isinstance(mu, list):
                mu = [mu for _ in range(n_dim)]

            X_.append(np.random.multivariate_normal(mu,sigma,size=n))

        else:
            X_.append(np.random.normal(mu,sigma,size=(n,n_dim)))

    return  np.concatenate([X_]).reshape(-1,n_dim)

def plot_clusters_2d(X, mus, sigmas=None, zoom=1, fig_kwargs={}, path=None):
    """
    Args:
        X (np.array): Data
        mus (np.array): (n_clusters, n_dim) means vectors
        sigmas (np.array): (n_clusters, n_dim, n_dim) variance matrices
        zoom (float, int): if > 1 enhances the size of elipses
        path (str): if and where to save figure
    """
    plt.figure(**fig_kwargs)
    clusters_col = ["red", "blue", "yellow", "purple", "black"] * 100
    n_clusters = mus.shape[0]
    if sigmas is None:
        sigmas = np.array([np.identity(X.shape[1])]*n_clusters)

    fig, a = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=(10,10))
    a.scatter(X[:,0], X[:,1])
    a.scatter(mus[:,0].reshape(-1), mus[:,1].reshape(-1), 
              c=clusters_col[:n_clusters], marker='x')

    for k in range(n_clusters):
        w, v = np.linalg.eig(sigmas[k,:,:])
        e = Ellipse(xy=tuple(mus[k,:]),
                width=zoom*w[0], 
                height=zoom*w[1],
                angle=np.arccos((v[0,:].T @ np.matrix([[1,0]]).T).item()/ (np.linalg.norm(v[:,0]))))
        e.set_clip_box(a.bbox)
        e.set_alpha(0.5)
        e.set_facecolor(clusters_col[k])
        a.add_artist(e)
        
    if not path is None:
        plt.savefig(path)

    plt.show()
