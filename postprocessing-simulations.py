# Databricks notebook source
# MAGIC %sh
# MAGIC ls /dbfs/mnt/gti/GEEN/results

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error

# COMMAND ----------

from scipy.stats import gaussian_kde
from sklearn.preprocessing import minmax_scale
import numpy as np
import scipy.stats as stats

def generate_scatter_size(x, y):
    xy = np.append(x, y, axis=1)
    kde = stats.kde.gaussian_kde(xy.T)
    s = kde(xy.T)
    s = (minmax_scale(s) + 0.1) * 10
    return s

# COMMAND ----------

result_folder = "/dbfs/mnt/gti/GEEN/results"
simulation_suffix = '-discrete_xstar-x4_distinct'

# COMMAND ----------

# DBTITLE 1,sample data plot for baseline simulation
baseline_data = pd.read_csv(os.path.join(result_folder, 'Basement' +simulation_suffix + '_simulation_res_data.csv'))

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, constrained_layout=True, figsize=(3.2, 3.2), dpi=300)
fig.suptitle('Data in Training Sample', fontsize=9)
axs[0,0].scatter(x=baseline_data.x_star, y=baseline_data.x1, s=1)
axs[0,0].set_title('$X^{1}$ ', fontsize=7, x=0.1, y=0.8)
axs[0,1].scatter(x=baseline_data.x_star, y=baseline_data.x2, s=1)
axs[0,1].set_title('$X^{2}$ ', fontsize=7, x=0.8, y=0.8)
axs[1,0].scatter(x=baseline_data.x_star, y=baseline_data.x3, s=1)
axs[1,0].set_title('$X^{3}$ ', fontsize=7, x=0.5, y=0.8)
axs[1,0].set_xlabel('$X^{*}$', fontsize=5)
axs[1,1].scatter(x=baseline_data.x_star, y=baseline_data.x4, s=1)
axs[1,1].set_title('$X^{4}$', fontsize=7, x=0.5, y=0.8)
axs[1,1].set_xlabel('$X^{*}$', fontsize=5)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=5 )
    ax.tick_params(axis='y', labelsize=5 )
#fig.tight_layout()
fig.savefig(os.path.join(result_folder, 'baseline' + simulation_suffix + '_training_samples.png'), dpi=fig.dpi)
plt.show()

print('rmse of x1 is %f'%mean_squared_error(baseline_data.x1, baseline_data.x_star, squared=False))
print(np.corrcoef(baseline_data.x1, baseline_data.x_star)[0,1])
print(np.corrcoef(baseline_data.x2, baseline_data.x_star)[0,1])
print(np.corrcoef(baseline_data.x3, baseline_data.x_star)[0,1])
print(np.corrcoef(baseline_data.x4, baseline_data.x_star)[0,1])

# COMMAND ----------

# DBTITLE 1,sample data plot for linear error simulation
linearError_data = pd.read_csv(os.path.join(result_folder, 'linearError' + simulation_suffix + '_simulation_res_data.csv'))

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, constrained_layout=True, figsize=(3.2, 3.2), dpi=300)
fig.suptitle('Data in Training Sample', fontsize=9)
axs[0,0].scatter(x=linearError_data.x_star, y=linearError_data.x1, s=1)
axs[0,0].set_title('$X^{1}$ ', fontsize=7, x=0.1, y=0.8)
axs[0,1].scatter(x=linearError_data.x_star, y=linearError_data.x2, s=1)
axs[0,1].set_title('$X^{2}$ ', fontsize=7, x=0.8, y=0.8)
axs[1,0].scatter(x=linearError_data.x_star, y=linearError_data.x3, s=1)
axs[1,0].set_title('$X^{3}$ ', fontsize=7, x=0.5, y=0.8)
axs[1,0].set_xlabel('$X^{*}$', fontsize=5)
axs[1,1].scatter(x=linearError_data.x_star, y=linearError_data.x4, s=1)
axs[1,1].set_title('$X^{4}$', fontsize=7, x=0.5, y=0.8)
axs[1,1].set_xlabel('$X^{*}$', fontsize=5)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=5 )
    ax.tick_params(axis='y', labelsize=5 )
#fig.tight_layout()
fig.savefig(os.path.join(result_folder, 'linearError' + simulation_suffix + '_training_samples.png'), dpi=fig.dpi)
plt.show()

print('rmse of x1 is %f'%mean_squared_error(linearError_data.x1, linearError_data.x_star, squared=False))
print(np.corrcoef(linearError_data.x1, linearError_data.x_star)[0,1])
print(np.corrcoef(linearError_data.x2, linearError_data.x_star)[0,1])
print(np.corrcoef(linearError_data.x3, linearError_data.x_star)[0,1])
print(np.corrcoef(linearError_data.x4, linearError_data.x_star)[0,1])

# COMMAND ----------

# DBTITLE 1,sample data plot for double error simulation
doubleError_data = pd.read_csv(os.path.join(result_folder, 'doubleError' +simulation_suffix + '_simulation_res_data.csv'))

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, constrained_layout=True, figsize=(3.2, 3.2), dpi=300)
fig.suptitle('Data in Training Sample', fontsize=9)
axs[0,0].scatter(x=doubleError_data.x_star, y=doubleError_data.x1, s=1)
axs[0,0].set_title('$X^{1}$ ', fontsize=7, x=0.1, y=0.8)
axs[0,1].scatter(x=doubleError_data.x_star, y=doubleError_data.x2, s=1)
axs[0,1].set_title('$X^{2}$ ', fontsize=7, x=0.8, y=0.8)
axs[1,0].scatter(x=doubleError_data.x_star, y=doubleError_data.x3, s=1)
axs[1,0].set_title('$X^{3}$ ', fontsize=7, x=0.5, y=0.8)
axs[1,0].set_xlabel('$X^{*}$', fontsize=5)
axs[1,1].scatter(x=doubleError_data.x_star, y=doubleError_data.x4, s=1)
axs[1,1].set_title('$X^{4}$', fontsize=7, x=0.5, y=0.8)
axs[1,1].set_xlabel('$X^{*}$', fontsize=5)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=5 )
    ax.tick_params(axis='y', labelsize=5 )
#fig.tight_layout()
fig.savefig(os.path.join(result_folder, 'doubleError' + simulation_suffix + '_training_samples.png'), dpi=fig.dpi)
plt.show()

print('rmse of x1 is %f'%mean_squared_error(doubleError_data.x1, doubleError_data.x_star, squared=False))
print(np.corrcoef(doubleError_data.x1, doubleError_data.x_star)[0,1])
print(np.corrcoef(doubleError_data.x2, doubleError_data.x_star)[0,1])
print(np.corrcoef(doubleError_data.x3, doubleError_data.x_star)[0,1])
print(np.corrcoef(doubleError_data.x4, doubleError_data.x_star)[0,1])

# COMMAND ----------

import re
from sklearn.cluster import KMeans
if re.search('-discrete_xstar', simulation_suffix, re.I):
    init = np.stack([baseline_data[baseline_data.x_star==i].sample(n=1)[['x1','x2','x3', 'x4']].values.flatten() for i in range(11)], axis=0)
    baseline_kmeans = KMeans(n_clusters=11, init=init).fit(baseline_data[['x1','x2','x3', 'x4']])
    baseline_data['kmeans'] = baseline_kmeans.labels_
    init = np.stack([linearError_data[linearError_data.x_star==i].sample(n=1)[['x1','x2','x3', 'x4']].values.flatten() for i in range(11)], axis=0)
    linearError_kmeans = KMeans(n_clusters=11, init=init).fit(linearError_data[['x1','x2','x3', 'x4']])
    linearError_data['kmeans'] = linearError_kmeans.labels_
    init = np.stack([doubleError_data[doubleError_data.x_star==i].sample(n=1)[['x1','x2','x3', 'x4']].values.flatten() for i in range(11)], axis=0)
    doubleError_kmeans = KMeans(n_clusters=11, init=init).fit(doubleError_data[['x1','x2','x3', 'x4']])
    doubleError_data['kmeans'] = doubleError_kmeans.labels_

    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, constrained_layout=True, figsize=(3.2, 3.2), dpi=300)
    #fig.suptitle('Result', fontsize=9)
    axs[0,0].scatter(x=baseline_data.x_star.values[-1000:], y=baseline_data.kmeans.values[-1000:], s=1)
    axs[0,0].set_title('Baseline', fontsize=7, x=0.3, y=0.8)
    axs[0,0].set_ylabel('kmeans $X^{*}$', fontsize=5)
    axs[0,0].set_xlabel('$X^{*}$', fontsize=5)
    axs[0,1].scatter(x=linearError_data.x_star.values[-1000:], y=linearError_data.kmeans.values[-1000:], s=1)
    axs[0,1].set_title('Linear Error', fontsize=7, x=0.3, y=0.8)
    axs[0,1].set_xlabel('$X^{*}$', fontsize=5)
    axs[1,0].scatter(x=doubleError_data.x_star.values[-1000:], y=doubleError_data.kmeans.values[-1000:], s=1)
    axs[1,0].set_title('Double Error', fontsize=7, x=0.3, y=0.8)
    axs[1,0].set_xlabel('$X^{*}$', fontsize=5)
    axs[1,0].set_ylabel('kmeans $X^{*}$', fontsize=5)


    for ax in axs.flat:
        ax.tick_params(axis='x', labelsize=5 )
        ax.tick_params(axis='y', labelsize=5 )
    #fig.tight_layout()
    fig.delaxes(axs[1][1])
    fig.savefig(os.path.join(result_folder, 'results_kmeans' + simulation_suffix + '.png'), dpi=fig.dpi)
    print(np.corrcoef(baseline_data.kmeans[-1000:], baseline_data.x_star[-1000:])[0,1])
    print(np.corrcoef(linearError_data.kmeans[-1000:], linearError_data.x_star[-1000:])[0,1])
    print(np.corrcoef(doubleError_data.kmeans[-1000:], doubleError_data.x_star[-1000:])[0,1])

# COMMAND ----------

fig, axs = plt.subplots(
    nrows=2, ncols=2, sharey=True, constrained_layout=True, figsize=(3.2, 3.2), dpi=300
)
# fig.suptitle('Result', fontsize=9)
if re.search("-discrete_xstar", simulation_suffix, re.I):
    s = generate_scatter_size(
        baseline_data.x_star.values[-1000:].reshape(1000, 1),
        baseline_data.x_star_gen.values[-1000:].reshape(1000, 1),
    )
else:
    s = 1
axs[0, 0].scatter(
    x=baseline_data.x_star.values[-1000:],
    y=baseline_data.x_star_gen.values[-1000:],
    s=s,
)
axs[0, 0].set_title("Baseline", fontsize=7, x=0.3, y=0.8)
axs[0, 0].set_ylabel("Generated $X^{*}$", fontsize=5)
axs[0, 0].set_xlabel("$X^{*}$", fontsize=5)
if re.search("-discrete_xstar", simulation_suffix, re.I):
    s = generate_scatter_size(
        linearError_data.x_star.values[-1000:].reshape(1000, 1),
        linearError_data.x_star_gen.values[-1000:].reshape(1000, 1),
    )
else:
    s = 1
axs[0, 1].scatter(
    x=linearError_data.x_star.values[-1000:],
    y=linearError_data.x_star_gen.values[-1000:],
    s=s,
)
axs[0, 1].set_title("Linear Error", fontsize=7, x=0.3, y=0.8)
axs[0, 1].set_xlabel("$X^{*}$", fontsize=5)
if re.search("-discrete_xstar", simulation_suffix, re.I):
    s = generate_scatter_size(
        doubleError_data.x_star.values[-1000:].reshape(1000, 1),
        doubleError_data.x_star_gen.values[-1000:].reshape(1000, 1),
    )
else:
    s = 1
axs[1, 0].scatter(
    x=doubleError_data.x_star.values[-1000:],
    y=doubleError_data.x_star_gen.values[-1000:],
    s=1,
)
axs[1, 0].set_title("Double Error", fontsize=7, x=0.3, y=0.8)
axs[1, 0].set_xlabel("$X^{*}$", fontsize=5)
axs[1, 0].set_ylabel("Generated $X^{*}$", fontsize=5)
if re.search("-discrete_xstar", simulation_suffix, re.I):
    s = generate_scatter_size(
        baseline_data.x_star.values[-1000:].reshape(1000, 1),
        baseline_data.kmeans.values[-1000:].reshape(1000, 1),
    )
    axs[0, 0].scatter(
        x=baseline_data.x_star.values[-1000:],
        y=baseline_data.kmeans.values[-1000:],
        s=s[-1000:],
        c="r",
    )
    axs[0, 0].legend(['GEEN', 'k-mean'], loc='lower right', fontsize=5)
    s = generate_scatter_size(
        linearError_data.x_star.values[-1000:].reshape(1000, 1),
        linearError_data.kmeans.values[-1000:].reshape(1000, 1),
    )
    axs[0, 1].scatter(
        x=linearError_data.x_star.values[-1000:],
        y=linearError_data.kmeans.values[-1000:],
        s=s[-1000:],
        c="r",
    )
    axs[0, 1].legend(['GEEN', 'k-mean'], loc='lower right', fontsize=5)
    s = generate_scatter_size(
        doubleError_data.x_star.values[-1000:].reshape(1000, 1),
        doubleError_data.kmeans.values[-1000:].reshape(1000, 1),
    )
    axs[1, 0].scatter(
        x=doubleError_data.x_star.values[-1000:],
        y=doubleError_data.kmeans.values[-1000:],
        s=s[-1000:],
        c="r",
    )
    axs[1, 0].legend(['GEEN', 'k-mean'], loc='lower right', fontsize=5)


for ax in axs.flat:
    ax.tick_params(axis="x", labelsize=5)
    ax.tick_params(axis="y", labelsize=5)
# fig.tight_layout()
fig.delaxes(axs[1][1])
fig.savefig(
    os.path.join(result_folder, "results" + simulation_suffix + ".png"), dpi=fig.dpi
)

# COMMAND ----------

baseline_corr_data = pd.read_csv(os.path.join(result_folder, 'Basement'+simulation_suffix+'_simulation_corr.csv'))
linearError_corr_data = pd.read_csv(os.path.join(result_folder, 'linearError'+simulation_suffix+'_simulation_corr.csv'))
doubleError_corr_data = pd.read_csv(os.path.join(result_folder, 'doubleError'+simulation_suffix+'_simulation_corr.csv'))

fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, constrained_layout=True, figsize=(3.2, 3.2), dpi=300)
fig.suptitle('Distribution of Correlations in Test Sample', fontsize=9)
axs[0,0].hist(baseline_corr_data['corr'].values, bins=50)
axs[0,0].set_title('Baseline', fontsize=7, x=0.3, y=0.8)
axs[0,1].hist(linearError_corr_data['corr'].values, bins=50)
axs[0,1].set_title('Linear Error', fontsize=7, x=0.3, y=0.8)
axs[1,0].hist(doubleError_corr_data['corr'].values, bins=50)
axs[1,0].set_title('Double Error', fontsize=7, x=0.3, y=0.8)

#fig.tight_layout()
fig.delaxes(axs[1][1])
fig.savefig(os.path.join(result_folder, 'results' + simulation_suffix + '_corr.png'), dpi=fig.dpi)

print(baseline_corr_data['corr'].min(), baseline_corr_data['corr'].max(), baseline_corr_data['corr'].median())
print(linearError_corr_data['corr'].min(), linearError_corr_data['corr'].max(), linearError_corr_data['corr'].median())
print(doubleError_corr_data['corr'].min(), doubleError_corr_data['corr'].max(), doubleError_corr_data['corr'].median())

# COMMAND ----------


