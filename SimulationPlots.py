import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from Transforms import logistic
np_logistic = np.vectorize(logistic)

def run_simulation_noise_after(depths, linear_model, linear_model_scaled, max_property,
                   std_dev_surface, std_dev_slope,
                   logistic_std_dev_surface, logistic_std_dev_slope, logistic_std_dev_range, logisitic_std_dev_min):
    zs = []
    property_values = []
    property_logistic = []
    property_constant = []

    trend = np_logistic(np.array([linear_model.params.depth * z + linear_model.params.const for z in depths]))
    for _ in range(100):
        zs += depths
        noise = np.array([np.random.normal(scale=std_dev_surface + z * std_dev_slope) for z in depths])
        noise_logisitic = np.array(
            [np.random.normal(scale=logistic(logistic_std_dev_surface + z * logistic_std_dev_slope)
                                    * logistic_std_dev_range + logisitic_std_dev_min) for z in depths])
        noise_fixed = np.random.normal(scale=0.05, size=400)

        signal = trend + noise
        signal_logistic = trend + noise_logisitic
        signal_constant = trend + noise_fixed

        property_values += signal.tolist()
        property_logistic += signal_logistic.tolist()
        property_constant += signal_constant.tolist()

    cell_count = len(property_values)
    unscaled_hetero_invalid_count = len([p for p in property_values if p <= 0 or p >= 1])
    unscaled_logistic_invalid_count = len([p for p in property_logistic if p <= 0 or p >= 1])
    unscaled_homo_invalid_count = len([p for p in property_constant if p <= 0 or p >= 1])

    property_values = np.array(property_values)
    property_logistic = np.array(property_logistic)
    property_constant = np.array(property_constant)

    invalid_property_indices = np.array([p <= 0 or p >= 1 for p in property_values])
    invalid_property_logistic_indices = np.array([p <= 0 or p >= 1 for p in property_logistic])
    invalid_property_constant_indices = np.array([p <= 0 or p >= 1 for p in property_constant])
    zs = np.array(zs)

    df_property = pd.DataFrame({'depth': zs, 'property': property_values})
    df_property_logistic = pd.DataFrame({'depth': zs, 'property': property_logistic})
    df_property_constant = pd.DataFrame({'depth': zs, 'property': property_constant})
    grouped_property = df_property.groupby('depth')
    grouped_property_logistic = df_property_logistic.groupby('depth')
    grouped_property_constant = df_property_constant.groupby('depth')

    property_mean = grouped_property.mean()
    property_std = grouped_property.std()
    fig, [[ax7, ax8, ax9], [ax, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(3, 3, figsize=(20, 20))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_title("Property simulation with linearly varying std. dev.")
    ax.set_xlabel("p")
    ax.set_ylabel("Depth")
    ax.scatter(property_values[~invalid_property_indices], zs[~invalid_property_indices])
    ax.scatter(property_values[invalid_property_indices], zs[invalid_property_indices])
    ax.plot(property_mean, property_mean.index, 'k')
    ax.plot(property_mean + property_std, property_mean.index, 'b')
    ax.plot(property_mean - property_std, property_mean.index, 'b')
    ax.plot(property_mean + 4 * property_std, property_mean.index, 'r')
    ax.plot(property_mean - 4 * property_std, property_mean.index, 'r')

    property_mean = grouped_property_logistic.mean()
    property_std = grouped_property_logistic.std()
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    ax2.set_title("Property simulation with logistically varying std. dev.")
    ax2.set_xlabel("p")
    ax2.set_ylabel("Depth")
    ax2.scatter(property_logistic[~invalid_property_logistic_indices], zs[~invalid_property_logistic_indices])
    ax2.scatter(property_logistic[invalid_property_logistic_indices], zs[invalid_property_logistic_indices])
    ax2.plot(property_mean, property_mean.index, 'k')
    ax2.plot(property_mean + property_std, property_mean.index, 'b')
    ax2.plot(property_mean - property_std, property_mean.index, 'b')
    ax2.plot(property_mean + 4 * property_std, property_mean.index, 'r')
    ax2.plot(property_mean - 4 * property_std, property_mean.index, 'r')

    property_mean = grouped_property_constant.mean()
    property_std = grouped_property_constant.std()
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')
    ax3.set_title("Property simulation with fixed std. dev.")
    ax3.set_xlabel("p")
    ax3.set_ylabel("Depth")
    ax3.scatter(property_constant[~invalid_property_constant_indices], zs[~invalid_property_constant_indices])
    ax3.scatter(property_constant[invalid_property_constant_indices], zs[invalid_property_constant_indices])
    ax3.plot(property_mean, property_mean.index, 'k')
    ax3.plot(property_mean + property_std, property_mean.index, 'b')
    ax3.plot(property_mean - property_std, property_mean.index, 'b')
    ax3.plot(property_mean + 4 * property_std, property_mean.index, 'r')
    ax3.plot(property_mean - 4 * property_std, property_mean.index, 'r');

    zs = []
    property_values = []
    property_logistic = []
    property_constant = []
    trend = np_logistic(np.array(
        [linear_model_scaled.params.depth * z + linear_model_scaled.params.const for z in depths])) * max_property
    for _ in range(100):
        zs += depths
        noise = np.array([np.random.normal(scale=std_dev_surface + z * std_dev_slope) for z in depths])
        noise_logisitic = np.array(
            [np.random.normal(scale=logistic(logistic_std_dev_surface + z * logistic_std_dev_slope)
                                * logistic_std_dev_range + logisitic_std_dev_min) for z in depths])
        noise_fixed = np.random.normal(scale=0.05, size=400)

        signal = trend + noise
        signal_logistic = trend + noise_logisitic
        signal_constant = trend + noise_fixed

        property_values += signal.tolist()
        property_logistic += signal_logistic.tolist()
        property_constant += signal_constant.tolist()

    property_values = np.array(property_values)
    property_logistic = np.array(property_logistic)
    property_constant = np.array(property_constant)
    zs = np.array(zs)
    invalid_property_indices = np.array([p <= 0 or p >= max_property for p in property_values])
    invalid_property_logistic_indices = np.array([p <= 0 or p >= max_property for p in property_logistic])
    invalid_property_constant_indices = np.array([p <= 0 or p >= max_property for p in property_constant])
    scaled_hetero_invalid_count = len([p for p in property_values if p <= 0 or p >= max_property])
    scaled_logistic_invalid_count = len([p for p in property_logistic if p <= 0 or p >= max_property])
    scaled_homo_invalid_count = len([p for p in property_constant if p <= 0 or p >= max_property])

    df_property = pd.DataFrame({'depth': zs, 'property': property_values})
    df_property_logistic = pd.DataFrame({'depth': zs, 'property': property_logistic})
    df_property_constant = pd.DataFrame({'depth': zs, 'property': property_constant})
    grouped_property = df_property.groupby('depth')
    grouped_property_logistic = df_property_logistic.groupby('depth')
    grouped_property_constant = df_property_constant.groupby('depth')
    property_mean = grouped_property.mean()
    property_std = grouped_property.std()
    ax4.xaxis.tick_top()
    ax4.xaxis.set_label_position('top')
    ax4.set_title("Property scaled simulation with varying std. dev.")
    ax4.set_xlabel("p")
    ax4.set_ylabel("Depth")
    ax4.scatter(property_values[~invalid_property_indices], zs[~invalid_property_indices])
    ax4.scatter(property_values[invalid_property_indices], zs[invalid_property_indices])
    ax4.plot(property_mean, property_mean.index, 'k')
    ax4.plot(property_mean + property_std, property_mean.index, 'b')
    ax4.plot(property_mean - property_std, property_mean.index, 'b')
    ax4.plot(property_mean + 4 * property_std, property_mean.index, 'r')
    ax4.plot(property_mean - 4 * property_std, property_mean.index, 'r')

    property_mean = grouped_property_logistic.mean()
    property_std = grouped_property_logistic.std()
    ax5.xaxis.tick_top()
    ax5.xaxis.set_label_position('top')
    ax5.set_title("Property simulation with logistically varying std. dev.")
    ax5.set_xlabel("p")
    ax5.set_ylabel("Depth")
    ax5.scatter(property_logistic[~invalid_property_logistic_indices], zs[~invalid_property_logistic_indices])
    ax5.scatter(property_logistic[invalid_property_logistic_indices], zs[invalid_property_logistic_indices])
    ax5.plot(property_mean, property_mean.index, 'k')
    ax5.plot(property_mean + property_std, property_mean.index, 'b')
    ax5.plot(property_mean - property_std, property_mean.index, 'b')
    ax5.plot(property_mean + 4 * property_std, property_mean.index, 'r')
    ax5.plot(property_mean - 4 * property_std, property_mean.index, 'r')

    property_mean = grouped_property_constant.mean()
    property_std = grouped_property_constant.std()
    ax6.xaxis.tick_top()
    ax6.xaxis.set_label_position('top')
    ax6.set_title("Property scaled simulation with fixed std. dev.")
    ax6.set_xlabel("p")
    ax6.set_ylabel("Depth")
    ax6.scatter(property_constant[~invalid_property_constant_indices], zs[~invalid_property_constant_indices])
    ax6.scatter(property_constant[invalid_property_constant_indices], zs[invalid_property_constant_indices])
    ax6.plot(property_mean, property_mean.index, 'k')
    ax6.plot(property_mean + property_std, property_mean.index, 'b')
    ax6.plot(property_mean - property_std, property_mean.index, 'b')
    ax6.plot(property_mean + 4 * property_std, property_mean.index, 'r')
    ax6.plot(property_mean - 4 * property_std, property_mean.index, 'r')
    ax7.xaxis.tick_top()
    ax7.xaxis.set_label_position('top')
    ax7.set_title("Heteroscedastic model (linear)")
    ax7.set_xlabel("$\sigma$")
    ax7.set_ylabel("Depth")
    ax7.plot(depths, [std_dev_slope * z + std_dev_surface for z in depths])
    ax8.xaxis.tick_top()
    ax8.xaxis.set_label_position('top')
    ax8.set_title("Heteroscedastic model (logistic)")
    ax8.set_xlabel("$\sigma$")
    ax8.set_ylabel("Depth")
    ax8.plot(depths, [logistic(logistic_std_dev_slope * z + logistic_std_dev_surface)
                      * logistic_std_dev_range + logisitic_std_dev_min for z in depths])
    ax9.xaxis.tick_top()
    ax9.xaxis.set_label_position('top')
    ax9.set_title("Hemoscedastic model")
    ax9.set_xlabel("$\sigma$")
    ax9.set_ylabel("Depth")
    ax9.plot(depths, [0.05 for _ in depths])
    fig.tight_layout()

    return cell_count, \
           unscaled_hetero_invalid_count, \
           unscaled_logistic_invalid_count, \
           unscaled_homo_invalid_count, \
           scaled_hetero_invalid_count, \
           scaled_logistic_invalid_count, \
           scaled_homo_invalid_count

def run_simulation_noise_before(depths, linear_model, linear_model_scaled, max_property,
                                logit_std_dev_surface, logit_std_dev_slope):
    zs = []
    property_values = []
    property_constant = []

    logit_property_values = []
    logit_property_constant = []

    trend = np.array([linear_model.params.depth * z + linear_model.params.const for z in depths])
    for _ in range(100):
        zs += depths
        noise = np.array([np.random.normal(scale=logit_std_dev_surface + z * logit_std_dev_slope) for z in depths])
        noise_fixed = np.random.normal(scale=0.35, size=400)

        signal = trend + noise
        signal_constant = trend + noise_fixed

        property_values += np_logistic(signal).tolist()
        property_constant += (np_logistic(signal_constant)).tolist()

        logit_property_values += (signal).tolist()
        logit_property_constant += (signal_constant).tolist()

    df_property = pd.DataFrame({'depth': zs, 'property': logit_property_values})
    df_property_constant = pd.DataFrame({'depth': zs, 'property': logit_property_constant})
    grouped_property = df_property.groupby('depth')
    grouped_property_constant = df_property_constant.groupby('depth')

    property_mean = grouped_property.mean()
    property_std = grouped_property.std()

    fig, [[ax, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 20))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_title("Property simulation with varying std. dev.")
    ax.set_xlabel("p")
    ax.set_ylabel("Depth")
    ax.scatter(property_values, zs)
    ax.plot(np_logistic(property_mean), property_mean.index, 'k')
    ax.plot(np_logistic(property_mean + property_std), property_mean.index, 'b')
    ax.plot(np_logistic(property_mean - property_std), property_mean.index, 'b')
    ax.plot(np_logistic(property_mean + 4 * property_std), property_mean.index, 'r')
    ax.plot(np_logistic(property_mean - 4 * property_std), property_mean.index, 'r')

    property_mean = grouped_property_constant.mean()
    property_std = grouped_property_constant.std()

    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    ax2.set_title("Property simulation with fixed std. dev.")
    ax2.set_xlabel("p")
    ax2.set_ylabel("Depth")
    ax2.scatter(property_constant, zs)
    ax2.plot(np_logistic(property_mean), property_mean.index, 'k')
    ax2.plot(np_logistic(property_mean + property_std), property_mean.index, 'b')
    ax2.plot(np_logistic(property_mean - property_std), property_mean.index, 'b')
    ax2.plot(np_logistic(property_mean + 4 * property_std), property_mean.index, 'r')
    ax2.plot(np_logistic(property_mean - 4 * property_std), property_mean.index, 'r')

    zs = []
    property_values = []
    property_constant = []

    logit_property_values = []
    logit_property_constant = []

    trend = np.array([linear_model_scaled.params.depth * z + linear_model_scaled.params.const for z in depths])
    for _ in range(100):
        zs += depths
        noise = np.array([np.random.normal(scale=logit_std_dev_surface + z * logit_std_dev_slope) for z in depths])
        noise_fixed = np.random.normal(scale=0.35, size=400)

        signal = trend + noise
        signal_constant = trend + noise_fixed

        property_values += (np_logistic(signal) * max_property).tolist()
        property_constant += (np_logistic(signal_constant) * max_property).tolist()

        logit_property_values += (signal).tolist()
        logit_property_constant += (signal_constant).tolist()

    df_property = pd.DataFrame({'depth': zs, 'property': logit_property_values})
    df_property_constant = pd.DataFrame({'depth': zs, 'property': logit_property_constant})
    grouped_property = df_property.groupby('depth')
    grouped_phit_constant = df_property_constant.groupby('depth')

    property_mean = grouped_property.mean()
    property_std = grouped_property.std()

    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')
    ax3.set_title("Property scaled simulation with varying std. dev.")
    ax3.set_xlabel("p")
    ax3.set_ylabel("Depth")
    ax3.scatter(property_values, zs)
    ax3.plot(np_logistic(property_mean) * max_property, property_mean.index, 'k')
    ax3.plot(np_logistic(property_mean + property_std) * max_property, property_mean.index, 'b')
    ax3.plot(np_logistic(property_mean - property_std) * max_property, property_mean.index, 'b')
    ax3.plot(np_logistic(property_mean + 4 * property_std) * max_property, property_mean.index, 'r')
    ax3.plot(np_logistic(property_mean - 4 * property_std) * max_property, property_mean.index, 'r')

    property_mean = grouped_phit_constant.mean()
    property_std = grouped_phit_constant.std()

    ax4.xaxis.tick_top()
    ax4.xaxis.set_label_position('top')
    ax4.set_title("$\phi_T$ scaled simulation with fixed std. dev.")
    ax4.set_xlabel("$\phi_T$")
    ax4.set_ylabel("Depth")
    ax4.scatter(property_constant, zs)
    ax4.plot(np_logistic(property_mean) * max_property, property_mean.index, 'k')
    ax4.plot(np_logistic(property_mean + property_std) * max_property, property_mean.index, 'b')
    ax4.plot(np_logistic(property_mean - property_std) * max_property, property_mean.index, 'b')
    ax4.plot(np_logistic(property_mean + 4 * property_std) * max_property, property_mean.index, 'r')
    ax4.plot(np_logistic(property_mean - 4 * property_std) * max_property, property_mean.index, 'r')
    fig.tight_layout()