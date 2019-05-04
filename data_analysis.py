import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, FastICA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score


def load_features():
    df = pd.read_csv('final_ftr_matrix.csv')
    return df.drop(columns=['patient_nbr'])


def process_data(df):
    # Process data so that the given data matrix will be without any medication features.
    column_dict = {colname: colindex for colindex, colname in enumerate(df.columns)}
    meds = df[[col for col in df.columns if column_dict[col] in range(86, 153)]]
    rest_of_features = df.drop(columns=[col for col in df.columns if column_dict[col]
                                        in range(86, 153)] + [
                                           'num_medications', 'diabetesMed', 'change', 'readmitted'])
    return column_dict, meds, rest_of_features


def scale_features(ftr_mx):
    # Normalize features by dividing every feature in the maximal feature value.
    x = ftr_mx.values
    x_scaled = np.zeros(shape=x.shape)
    for feature in range(x.shape[1]):
        ftr_values = np.unique(x[:, feature])
        if np.array_equal(ftr_values, np.array([0, 1])):
            x_scaled[:, feature] = x[:, feature]
        else:
            scaler = MinMaxScaler()
            # scaler = StandardScaler()
            x_scaled[:, feature] = scaler.fit_transform(x[:, feature].reshape((-1, 1))).reshape((x_scaled.shape[0],))
    normed_rest_of_features = pd.DataFrame(x_scaled, columns=ftr_mx.columns)
    return normed_rest_of_features


def decompose_pca(original_matrix, n_components):
    pca = PCA(n_components=n_components, svd_solver='full')
    after_pca_vectors = pca.fit_transform(original_matrix)
    print('Explained variance ratio by vector: ', pca.explained_variance_ratio_)
    return after_pca_vectors


def decompose_ica(original_matrix, n_components):
    ica = FastICA(n_components=n_components, whiten=True, max_iter=500)
    after_ica_vectors = ica.fit_transform(original_matrix)
    return after_ica_vectors


def plot_decomposed(after_decomp, name):
    # Plot the decomposed feature matrix.
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(after_decomp[:, 0], after_decomp[:, 1], after_decomp[:, 2])
    plt.title('First 3 components (out of %d) of feature matrix after PCA')
    plt.grid(True)
    # plt.savefig(name + '.tif')
    plt.show()


def k_means_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, tol=0.01)
    kmeans.fit(data)
    return kmeans.labels_


def spectral_clustering(data, n_clusters):
    spec = SpectralClustering(n_clusters=n_clusters, gamma=2.)
    spec.fit(data)
    return spec.labels_


def score_clustering(data, labels):
    return silhouette_score(data, labels)


def profiles_people_without_meds(df):
    columns_dict, meds, remaining_features = process_data(df)
    normed_remaining_features = scale_features(remaining_features)

    # Decompose the feature matrix, using PCA or ICA.
    after_pca = decompose_pca(normed_remaining_features, 3)
    plot_decomposed(after_pca, 'pca')
    # after_ica = decompose_ica(remaining_features, 3)
    # plot_decomposed(after_ica, 'ica')


    # Use clustering methods to associate every sample to its cluster.
    cluster_numbers = [i for i in range(2, 50)]
    sil_scores = []
    for cluster_number in cluster_numbers:
        labels = k_means_clustering(after_pca, cluster_number)
        sil_scores.append(score_clustering(after_pca, labels))
    plt.figure()
    plt.plot(cluster_numbers, sil_scores, 'ro')
    plt.title('Average silhouette scores for k-means')
    plt.grid(True)
    plt.savefig('Silhouette_k_means.tif')
    return None


if __name__ == "__main__":
    f = load_features()
    profiles_people_without_meds(f)

