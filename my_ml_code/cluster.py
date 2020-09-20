from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from kneed import KneeLocator


def eval_knn_traffic(kmeans, cols, verbose=True, plot=True):

    nc = len(kmeans.cluster_centers_)
    fig, ax = plt.subplots(nc, 1, figsize=(10, 2.2*nc))
    for ind, center in enumerate(kmeans.cluster_centers_):
        ax[ind].bar(cols, center, alpha=0.7)
        ax[ind].set_title(f'Centroid #{ind+1}')
        ax[ind].set_ylim(0, 0.5)
        ax[ind].set_yticks(np.arange(6)/10)
    fig.tight_layout()

    if verbose:
        print(f'Final Inertia: {kmeans.inertia_:.2f}\n')

        counts = pd.Series(kmeans.labels_).value_counts()
        print('----- Cluster Counts -----')
        for c in range(counts.shape[0]):
            print(f'Cluster #{c+1}: {counts.loc[c]}')
        print('--------------------------')

    return fig, ax


def knn_elbow(n_clusters, sse, plot=True, verbose=True):
    if plot:
        fig, ax = plt.subplots()
        ax.plot(n_clusters, sse)
        ax.set_xticks(n_clusters)
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("SSE")

    kl = KneeLocator(
        n_clusters, sse, curve="convex", direction="decreasing"
    )
    if verbose:
        print(f'Elbow Clusters: {kl.elbow}')
    return kl


def knn_sillouette(n_clusters, sil, plot=True, verbose=True):

    fig, ax = plt.subplots()
    ax.plot(n_clusters, sil)
    ax.set_xticks(n_clusters)
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Sillouette Coefficient")


def ag_plot_avg(X, AG, cols):
    ag_df = X.copy()
    ag_df['label_ag'] = AG.labels_
    centroids = ag_df.groupby('label_ag').mean().to_numpy()

    nc = centroids.shape[0]
    fig, ax = plt.subplots(nc, 1, figsize=(10, 2.2*nc))
    for ind, center in enumerate(centroids):
        ax[ind].bar(cols, center, alpha=0.7)
        ax[ind].set_title(f'Centroid #{ind+1}')
        ax[ind].set_ylim(0, 0.5)
        ax[ind].set_yticks(np.arange(6)/10)
    fig.tight_layout()

    return fig, ax
