import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
import tqdm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from assignment_one.src.aux import franke_function
from assignment_one.src.plotting import latexify, format_axes
from assignment_one.src.regression_class import OLS

if __name__ == '__main__':
    N = 1000
    polynomial_degrees = range(0, 11)

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    X, Y = np.meshgrid(x, y)
    cartesian_product = np.dstack((X, Y)).reshape(-1, 2)
    z_grid = franke_function(X, Y)
    z_grid_noise = z_grid + np.random.normal(0, 0.05, size=(z_grid.shape))
    z = z_grid.ravel()
    z_noise = z_grid_noise.ravel()
    mse_scores = []
    r2_scores = []
    mse_scores_noise = []
    r2_scores_noise = []
    for d in tqdm.tqdm(polynomial_degrees):
        design_matrix = sklearn.preprocessing.PolynomialFeatures(degree=d).fit_transform(cartesian_product)
        ols = OLS(design_matrix, z)
        ols_noise = OLS(design_matrix, z_noise)

        mse_scores.append(ols.mse())
        r2_scores.append(ols.r2())
        mse_scores_noise.append(ols_noise.mse())
        r2_scores_noise.append(ols_noise.r2())


    latexify(3)
    plt.semilogy(polynomial_degrees, mse_scores, ls='-', marker='o')
    plt.semilogy(polynomial_degrees, mse_scores_noise, ls='--', marker='^')
    plt.xlabel('$d$')
    plt.ylabel('$\\mathrm{MSE}(\\mathbf{y}, \\hat{\\mathbf{y}})$')
    plt.tight_layout()
    format_axes(ax=plt.gca())
    plt.savefig('../article/images/OLS_MSE_score.pdf')
    plt.clf()

    plt.plot(polynomial_degrees, r2_scores, ls='-', marker='o')
    plt.plot(polynomial_degrees, r2_scores_noise, ls='--', marker='^')
    plt.xlabel('$d$')
    plt.ylabel('$\\mathrm{R}^2(\\mathbf{y}, \\hat{\\mathbf{y}})$')
    plt.tight_layout()
    format_axes(ax=plt.gca())
    plt.savefig('../article/images/OLS_R2_score.pdf')
