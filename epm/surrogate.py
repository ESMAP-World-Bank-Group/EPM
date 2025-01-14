import chaospy
from chaospy import generate_samples
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import gams.transfer as gt
import sys
sys.path.append('..')
from utils import extract_epm_results, process_epmresults, read_plot_specs
from sklearn import linear_model as lm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
from itertools import product
import os


UNIT = {
    'demand': 'MW/yr',
    'hydro': '%',
    'import': 'GW',
    'hfoprice': '$/kWh',
    'batterycost': '$/kW'
}

SCALE_FACTOR = {
    'demand': 1,
    'hydro': 1,
    'import': 1,
    'hfoprice': 1,
    'batterycost': 2.08 * 1000
}

NAMES_PARAMETERS = {
    'demand': 'Demand growth rate',
    'hydro': 'Hydro availability factor',
    'import': 'Import capacity',
    'hfoprice': 'HFO price',
    'batterycost': 'Variation in battery cost',
}

class NamedPoly:
    """Dictionary-like wrapper for vector numpoly polynomials with names."""

    def __init__(self, poly, names):
        self.poly = poly
        self.names = list(names)
        self.mapping = {k: i for i, k in enumerate(self.names)}

    def __getitem__(self, attr):
        return self.poly[self.mapping[attr]]

    def __call__(self, args):
        return pd.DataFrame(self.poly(*args), index=self.names).squeeze()


class NamedJ:
    """Dictionary-like wrapper for joint random variable generator with names."""

    def __init__(self, distributions):
        self.J = self.J_from_dict(distributions.values())
        self.names = distributions.keys()
        self.mapping = {k: i for i, k in enumerate(self.names)}

    def __getitem__(self, attr):
        return self.J[self.mapping[attr]]

    def J_from_dict(self, values):
        DD = []
        for v in values:
            D = getattr(chaospy, v["type"])
            DD.append(D(*v["args"]))
        return chaospy.J(*DD)

    def sample(self, size=100, rule="halton", fmt=3):
        samples = self.J.sample(size=size, rule=rule).round(fmt)
        index = [f"{n}" for n in self.names]
        return pd.DataFrame(samples, index=index)


def multiindex2array(multiindex):
    return np.array([np.array(row).astype(float) for row in multiindex]).T

def multiindex2df(multiindex):
    return pd.DataFrame(multiindex2array(multiindex), index=multiindex.names)


def build_surrogate(order, distribution, train_set, sklearn_model=None):
    """Building surrogate model with orthogonal polynomial chaos expansion."""
    samples = multiindex2array(train_set.index)
    pce = chaospy.expansion.stieltjes(order, distribution.J)
    surrogate = chaospy.fit_regression(pce, samples, train_set.values, sklearn_model)
    variables = train_set.columns
    return NamedPoly(surrogate, variables)


def build_pce_prediction(model, samples):
    """Predicting outcome for given samples, using the fitted polynomial model."""
    prediction = model(samples.values).clip(lower=0.0)
    prediction.columns = pd.MultiIndex.from_frame(samples.astype(str).T)
    return prediction.T


def get_output(epm_dict, variables, parameters, year=2030):
    """Extracting output outcomes of interest."""
    capacity = epm_dict['pCapacityByFuel'].copy()
    capacity = capacity.loc[capacity.scenario != 'baseline']
    for param in parameters:
        capacity.loc[:, param] = capacity.apply(lambda row:
                                                next(
                                                    part for part in row['scenario'].split('_') if param in part).split(
                                                    param)[1].replace('p', '.'), axis=1)

    capacity = capacity.drop(columns=['zone']).loc[(capacity.year == year) & (capacity.fuel.isin(variables))]
    list_cols = list(parameters) + ['fuel']
    capacity = capacity.set_index(list_cols)[['value']].squeeze().unstack('fuel')
    capacity.columns.name = None
    return capacity


def calculate_sobol(surrogate, distribution, sobol="t", decimals=3):
    """
    Estimating Sobol indices.
    Args:
        surrogate (NamedPoly): Polynomial chaos expansion
        distribution (NamedJ): Distribution of uncertain variables
        sobol (string): Which Sobol method to use to compute Sobol indices.
            When sobol = 't', total effect sensitivity index
            When sobol = 'm', first order effect sensitivity index
            When sobol = 'm2', second order effect sensitivity index
        decimals (int): How to round the result

    Returns:

    """
    func = getattr(chaospy, f"Sens_{sobol}")
    sobol = func(surrogate.poly, distribution.J).round(decimals)
    return pd.DataFrame(sobol, index=distribution.names, columns=surrogate.names)


def calculate_errors(prediction, truth):
    """Calculate errors between prediction and truth."""
    kws = dict(multioutput="raw_values")
    # Align index types
    prediction.index = pd.MultiIndex.from_arrays([
        prediction.index.get_level_values(i).astype(truth.index.get_level_values(i).dtype)
        for i in range(prediction.index.nlevels)
    ], names=prediction.index.names)
    diff = prediction - truth
    return pd.concat(
        {
            "mape": diff.abs().mean() / truth.mean() * 100,
            "mae": diff.abs().mean(),
            "r2": pd.Series(r2_score(truth, prediction, **kws), index=truth.columns),
            "mse": pd.Series(
                mean_squared_error(truth, prediction, **kws), index=truth.columns
            ),
            "rmse": pd.Series(
                mean_squared_error(truth, prediction, **kws), index=truth.columns
            )
            ** 0.5,
        },
        axis=1,
    )


def plot_sobol_bar(sobol, relative=True, fn=None):
    sobol = sobol.copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    if relative: sobol = sobol / sobol.sum()

    sobol = sobol.mul(100).round()

    y_lim = sobol.sum(axis=0).max() * 1.1

    sobol.T.plot.bar(ax=ax, stacked=True)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.legend(bbox_to_anchor=(1, 0.5), ncol=1, frameon=False, title="Uncertainty")

    # plt.ylim([0,max(sobol.sum().max(), 100)])
    plt.ylim([0, y_lim])
    plt.ylabel("Sobol [%]")

    ax.tick_params(axis='both', which=u'both', length=0)
    plt.xticks(rotation=-30, ha='left')

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')

    plt.close()


def plot_sobol(data, fn=None):
    data = data.copy()

    fig, ax = plt.subplots(figsize=(4, 7))

    data = data.mul(100).round()

    sns.heatmap(
        data,
        ax=ax,
        square=True,
        cmap="Purples",
        vmax=100,
        vmin=0,
        annot=True,
        # fmt=".2f",
        cbar=False,
    )
    plt.ylabel("Inputs")
    plt.xlabel("Outputs")
    if fn is not None:
        plt.savefig(fn, bbox_inches="tight")

    plt.close()


def plot_1D_baseline(surrogate, variable, parameter, coords, distribution, fixed,
            fn=None):
    poly = surrogate[variable]
    symbol = f"q{distribution.mapping[parameter]}"
    all_q = set(poly.names)
    to_qindex = distribution.mapping
    fixed = {"q" + str(to_qindex[k]): v for k, v in fixed.items()}
    P = []
    for coord in coords:
        symvalues = {symbol: coord}  # imposing the parameter for which we are doing the plot to take as value the coordinate
        assert set(symvalues.keys()).union((set(fixed.keys()))) == all_q, "Not all input parameters specified!"

        zpoly = poly(**fixed)
        P.append(zpoly(**symvalues))
    P = np.array(P)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(coords, P, linewidth=1, label="baseline", color='black')

    plt.ylabel(f"{variable} [GW]", fontsize=10)
    plt.xlabel(f"{NAMES_PARAMETERS[parameter]}\n{UNIT[parameter]}", fontsize=10)
    plt.legend(frameon=False)

    plt.box(False)
    plt.grid(None)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')

    plt.close()

def plot_1D_quantiles(surrogate, variable, parameter, coords, distribution, sample=10000,
            fn=None):
    poly = surrogate[variable]
    symbol = f"q{distribution.mapping[parameter]}"
    percentiles = [5, 25, 50, 75, 95]

    P = []
    for coord in coords:
        symvalues = {symbol: coord}  # imposing the parameter for which we are doing the plot to take as value the coordinate
        P.append(chaospy.Perc(poly(**symvalues), percentiles, distribution.J, sample=sample))
    P = np.array(P)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(coords, P[:, 2], linewidth=1, label="Q50", color='black')
    ax.fill_between(coords, P[:, 1], P[:, 3], alpha=0.2, label="Q25/Q75", color='blue')
    ax.fill_between(coords, P[:, 0], P[:, 4], alpha=0.2, label="Q5/Q95", color='grey')

    plt.ylabel(f"{variable} [GW]", fontsize=10)
    plt.xlabel(f"{NAMES_PARAMETERS[parameter]}\n{UNIT[parameter]}", fontsize=10)
    plt.legend(frameon=False)

    plt.box(False)
    plt.grid(None)

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')

    plt.close()


def plot_2D(surrogate, distribution, variable, xname, yname, xsamples=(0.5, 1.5, 20), ysamples=(0.5, 1.5, 20),
            fixed=1, dataset=None, contour_handles=None, vmin=160, vmax=230, levels=25, fn=None):

    surrogate_var = surrogate[variable]

    # TODO substitute distribution since only used for variable mapping
    to_qindex = distribution.mapping
    all_q = set(surrogate_var.names)

    qx = "q" + str(to_qindex[xname])
    qy = "q" + str(to_qindex[yname])

    if isinstance(fixed, (float, int)):
        fixed = {qo: fixed for qo in all_q - {qx, qy}}
    elif isinstance(fixed, dict):
        fixed = {"q" + str(to_qindex[k]): v for k, v in fixed.items()}
    else:
        raise NotImplementedError("Fixed input parameters not properly specified.")

    assert set(fixed.keys()).union({qx, qy}) == all_q, "Not all input parameters specified!"

    zpoly = surrogate_var(**fixed)

    z = np.array([zpoly(**{qx: xsamples, qy: y}) for y in ysamples])

    fig, ax = plt.subplots(figsize=(6, 5))

    # plt.contourf(xs, ys, z, levels=contour_handles.levels)

    contour_handles = ax.contourf(xsamples, ysamples, z)

    # Add the colorbar
    cbar = plt.colorbar(contour_handles, ax=ax)
    cbar.ax.set_title(f"{variable} [GW]", pad=10)

    # Change the yticks and scale them
    yticks = ax.get_yticks()  # Get the current yticks
    ax.set_yticks(yticks)     # Keep the same tick positions
    ax.set_yticklabels([f"{ytick * SCALE_FACTOR[yname]:.2f}" for ytick in yticks])  # Update the labels

    xticks = ax.get_xticks()  # Get the current xticks
    ax.set_xticks(xticks)     # Keep the same tick positions
    ax.set_xticklabels([f"{xtick * SCALE_FACTOR[xname]:.2f}" for xtick in xticks])  # Update the labels

    # cbar = plt.colorbar(contour_handles)

    plt.xlabel(f"{NAMES_PARAMETERS[xname]} [{UNIT[xname]}]")
    plt.ylabel(f"{NAMES_PARAMETERS[yname]} [{UNIT[yname]}]")

    plt.box(False)
    plt.grid(None)

    if dataset is not None:
        df = dataset.reset_index().astype(float)
        x = df[f"{xname}-cost"]
        y = df[f"{yname}-cost"]
        # plt.scatter(x, y, marker='.', s=5, alpha=0.2, color='grey')
        plt.scatter(x , y , marker='.', s=5, alpha=0.2, color='grey')

    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')

    plt.close()


# def plot_2D(surrogate, variable, parameter_x, parameter_y, coords_grid, distribution, sample=10000,
#             fn=None, n_samples=10):
#     poly = surrogate[variable]
#     symbol_x = f"q{distribution.mapping[parameter_x]}"
#     symbol_y = f"q{distribution.mapping[parameter_y]}"
#     percentiles = [5, 25, 50, 75, 95]
#
#     P = []
#     for coord in coords_grid:
#         symvalues = {
#             symbol_x: coord[0],
#                      symbol_y: coord[1]
#                      }  # imposing the parameter for which we are doing the plot to take as value the coordinate
#         P.append(chaospy.Perc(poly(**symvalues), percentiles, distribution.J, sample=sample))
#     P = np.array(P)
#
#     # Reshape `P` into a grid to make plotting easier
#     P_reshaped = P.reshape(n_samples, n_samples, -1)
#
#     fig, ax = plt.subplots(figsize=(3, 3))
#     ax.plot(coords, P[:, 2], linewidth=1, label="Q50", color='black')
#     ax.fill_between(coords, P[:, 1], P[:, 3], alpha=0.2, label="Q25/Q75", color='blue')
#     ax.fill_between(coords, P[:, 0], P[:, 4], alpha=0.2, label="Q5/Q95", color='grey')
#
#     plt.ylabel(f"{variable} [GW]", fontsize=10)
#     plt.xlabel(f"{NAMES_PARAMETERS[parameter]}\n{UNIT[parameter]}", fontsize=10)
#     plt.legend(frameon=False)
#
#     plt.box(False)
#     plt.grid(None)
#
#     if fn is not None:
#         plt.savefig(fn, bbox_inches='tight')
#
#     plt.close()


def plot_1D_mp(variant, surrogate, distribution, folder):
    var, param = variant
    fr = distribution[param].lower[0]
    to = distribution[param].upper[0]

    fn = Path(folder) / Path(f'1D-{var}-{param}.png')
    plot_1D_quantiles(surrogate, var, param, np.linspace(fr, to, 10), distribution, sample=10000, fn=fn)


def plot_1D_baseline_mp(variant, surrogate, distribution, folder, fixed):
    var, param = variant
    fr = distribution[param].lower[0]
    to = distribution[param].upper[0]

    fn = Path(folder) / Path(f'1D-{var}-{param}--baseline.png')

    plot_1D_baseline(surrogate, var, param, np.linspace(fr, to, 10), distribution, fixed=fixed, fn=fn)


def plot_2D_mp(variant, surrogate, distribution, dataset, folder, n_samples, fixed):
    var, param_x, param_y = variant
    fr_x = distribution[param_x].lower[0]
    to_x = distribution[param_x].upper[0]

    fr_y = distribution[param_y].lower[0]
    to_y = distribution[param_y].upper[0]

    # Define the 2D grid of uncertain variables (coordinates)
    coords_1 = np.linspace(fr_x, to_x, n_samples)  # Values for the first uncertain variable
    coords_2 = np.linspace(fr_y, to_y, n_samples)  # Values for the second uncertain variable

    # # Create a meshgrid of the coordinates
    # coords_1_grid, coords_2_grid = np.meshgrid(coords_1, coords_2)
    # coords_grid = np.stack([coords_1_grid.ravel(), coords_2_grid.ravel()], axis=-1)

    fn = Path(folder) / Path(f'2D-{var}-{param_x}-{param_y}.png')
    plot_2D(surrogate, distribution, var, param_x, param_y, coords_1, coords_2, fixed=fixed, dataset=None, fn=fn)


def plot_error_vs_order(train_set, test_set, folder, save=False, sklearn=None, max_order=6, max_n=400, subset_params=None):
    """Plot to understand error trade-off based on polynomial order."""
    results = {}
    for o in range(0, max_order):
        print(o, end=" ")

        surrogate = build_surrogate(o, distribution, train_set[:max_n], sklearn)

        test_samples = multiindex2df(test_set.index)
        test_predictions = build_pce_prediction(surrogate, test_samples)
        errors = calculate_errors(test_predictions, test_set)
        if subset_params is not None:
            errors = errors.loc[errors.index.isin(subset_params),:]
        results[o] = errors

    df = pd.concat(results, axis=1)

    for measure in ["r2", "mape", "mae", "rmse"]:
        data = df.T.unstack(level=0).loc[measure].unstack().T

        # colors = [TECH_COLORS[c] for c in data.columns]
        #
        # data.columns = data.columns.map(NAMES)

        fig, ax = plt.subplots(figsize=(3.5, 3))

        data.plot(ax=ax)
        ax.legend(bbox_to_anchor=(1, 0.5), ncol=1, frameon=False, title="Outcome")
        plt.xlabel("order of polynomial")
        plt.ylabel(measure.upper())
        plt.title(f"{len(train_set[:max_n])} training samples")
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ylims = dict(r2=[0.0, 1.05], mape=[0, 30], mae=[0, 100], rmse=[0, 100])
        plt.ylim(ylims[measure])

        if save:
            fn = Path(folder) / Path(f'error-{measure}-vs-order-sklearn.png')
            plt.savefig(fn, bbox_inches='tight')

        plt.close()


if __name__ == '__main__':
    RESULTS_FOLDER = 'output/simulations_run_20250109_141827'

    uncertainties = {
        # 'demand': {
        #     'type': 'Uniform',
        #     'args': (8, 20)
        # },
        'import': {
            'type': 'Uniform',
            'args': (0, 40)
        },
        'hydro': {
            'type': 'Uniform',
            'args': (0.7, 1.3)
        },
        'hfoprice': {
            'type': 'Uniform',
            'args': (12, 18)
        },
        'batterycost': {
            'type': 'Uniform',
            'args': (-0.5, 0)
        }
    }

    distribution = NamedJ(uncertainties)

    # dict_specs = read_plot_specs()
    # epmresults, scenarios = extract_epm_results(RESULTS_FOLDER, subset_params=['pCapacityByFuel'])  # SCENARIO
    #
    # epm_dict = process_epmresults(epmresults, dict_specs, input=Path('input'))
    # list_years = [2028, 2030, 2033]
    # list_techs = ["Solar", "Battery Storage 2h", "Battery Storage 4h", "Battery Storage 8h", "Battery Storage 3h", "Oil", "Hydro MC", "Hydro RoR"]
    # for year in list_years:
    #     dataset = get_output(epm_dict, variables=list_techs, parameters=distribution.names, year=year)
    #     dataset.to_csv(Path(RESULTS_FOLDER) / Path(f'dataset{year}.csv'), float_format='%.3f')  # to save to avoid computing each time

    folder = RESULTS_FOLDER
    dataset = pd.read_csv(Path(folder) / Path('dataset2033.csv'), index_col=list(range(len(distribution.names))))

    if not os.path.exists(Path(folder) / Path('images')):
        os.mkdir(Path(folder) / Path('images'))

    folder_images = Path(folder) / Path('images')

    train_set, test_set = train_test_split(dataset, train_size=0.8, test_size=0.2)

    polynomial_degree = 3
    number_of_uncertainty = 2

    sklearn = lm.Lars(verbose=True, fit_intercept=False)
    surrogate = build_surrogate(polynomial_degree, distribution, train_set, sklearn_model=None)

    train_samples = multiindex2df(train_set.index)
    train_predictions = build_pce_prediction(surrogate, train_samples)

    test_samples = multiindex2df(test_set.index)
    test_predictions = build_pce_prediction(surrogate, test_samples)

    errors_train = calculate_errors(train_predictions, train_set).round(2)
    errors_test = calculate_errors(test_predictions, test_set).round(2)

    # Checking the accuracy of the model on some specific points
    # samples_accuracy = pd.DataFrame({'0': [11,20,1], '1': [11,66,1]}, index=['demand', 'import', 'hydro'])
    # predictions_accuracy = build_pce_prediction(surrogate, samples_accuracy)

    plot_error_vs_order(train_set, test_set, folder_images, save=True, sklearn=None, max_order=6, max_n=400, subset_params=['Solar'])

    fixed = {
        'hfoprice': 15,
        'batterycost': -0.25,
        'import': 27
    }
    plot_1D_baseline_mp(('Battery Storage 4h', 'hydro'), surrogate, distribution, fixed=fixed, folder=folder_images)

    order = ["Solar", "Battery Storage 2h", "Battery Storage 4h", 'Oil', 'Hydro RoR']
    print("Total order Sobol")
    sobol_t = calculate_sobol(surrogate, distribution, sobol='t')[order]
    plot_sobol(sobol_t, fn=folder_images / Path(f'sobol-t.png'))
    plot_sobol_bar(sobol_t, relative=False, fn=folder_images / Path(f'sobol-t-bar.png'))

    print("First order Sobol")
    sobol_m = calculate_sobol(surrogate, distribution, sobol='m')[order]
    plot_sobol(sobol_m, fn=folder_images / Path(f'sobol-m.png'))
    plot_sobol_bar(sobol_m, relative=False, fn=folder_images / Path(f'sobol-m-bar.png'))

    variant = ('Solar', 'import', 'hydro')
    fixed = {
        'hfoprice': 15,
        'batterycost': 0,
    }

    plot_2D_mp(variant, surrogate, distribution, dataset, folder_images, n_samples=10, fixed=fixed)

    variant = ('Battery Storage 4h', 'import', 'batterycost')
    fixed = {
        'hfoprice': 15,
        'hydro': 1,
    }

    plot_2D_mp(variant, surrogate, distribution, dataset, folder_images, n_samples=20, fixed=fixed)

    variant = ('Battery Storage 4h', 'hydro', 'batterycost')
    fixed = {
        'hfoprice': 15,
        'import': 27,
    }

    plot_2D_mp(variant, surrogate, distribution, dataset, folder_images, n_samples=40, fixed=fixed)

    variants = product(dataset.columns, distribution.names)
    for variant in variants:
        plot_1D_mp(variant, surrogate, distribution, folder_images)

