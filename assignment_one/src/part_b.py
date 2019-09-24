import matplotlib.pyplot as plt
import numpy as np

from assignment_one.src.auxiliary import franke_function, create_variance_table, perform_cross_validation
from assignment_one.src.plotting import latexify, format_axes
from assignment_one.src.regression_class import OLS


def plot_errors():
    latexify(3)
    plt.semilogy(polynomial_degrees, mse_s[:, 1], ls='-', marker='o')
    plt.xlabel('$d$')
    plt.ylabel('$\\mathrm{MSE}(\\mathbf{y}, \\hat{\\mathbf{y}})$')
    plt.tight_layout()
    format_axes(ax=plt.gca())
    plt.savefig('../article/images/OLS_MSE_score_crossval.pdf')
    plt.clf()
    plt.plot(polynomial_degrees, r2_s[:, 1], ls='-', marker='o')
    plt.xlabel('$d$')
    plt.ylabel('$\\mathrm{R}^2(\\mathbf{y}, \\hat{\\mathbf{y}})$')
    plt.tight_layout()
    format_axes(ax=plt.gca())
    plt.savefig('../article/images/OLS_R2_score_crossval.pdf')


if __name__ == '__main__':
    N = 1000

    polynomial_degrees = range(0, 6)
    cartesian_product = np.random.random(size=(N, 2))
    X, Y = np.meshgrid(cartesian_product[:, 0], cartesian_product[:, 1])
    z_grid = franke_function(X, Y)
    z_grid_noise = z_grid + np.random.normal(0, 0.05, size=(z_grid.shape))
    z = z_grid.ravel()
    z_noise = z_grid_noise.ravel()

    number_of_folds = 10

    mse_s, r2_s, beta, beta_variance, beta_std, feature_names = perform_cross_validation(cartesian_product, z_noise,
                                                                                         number_of_folds=number_of_folds,
                                                                                         Regressor=OLS)
    # plot_errors()
    variance_table = create_variance_table(beta_variance, feature_names, polynomial_degrees)
    latex_variance = variance_table.to_latex(float_format=r"num{{{:.3e}}}".format, na_rep='')
    # Illustration of how the variance increases with model complexity
    print(latex_variance)

    expectation_table = create_variance_table(beta, feature_names, polynomial_degrees)
    expectation_table = expectation_table.to_latex(float_format=r"num{{{:.3e}}}".format, na_rep='')
    print(expectation_table)
