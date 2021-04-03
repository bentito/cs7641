# Assignment 3 Clustering and Dimensionality Reduction
import sys
import logging
import timeit
from time import time

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from dtreeviz.trees import *
from numpy import save
from numpy import load
import pathlib
from sklearn.neural_network import MLPClassifier


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def do_nn(dataset, X_train, y_train, X_test, y_test, pca_components, ica_components, rand_components, lfm_components):
    if dataset == 'faces':
        nn_model = MLPClassifier(alpha=0.05, hidden_layer_sizes=(20,), learning_rate='adaptive', max_iter=500)
    if dataset == 'forest':
        nn_model = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(20,),
                                 learning_rate='adaptive', max_iter=500)
    print("** NN for unreduced/unprojected data set")
    nn_model.fit(X_train, y_train)
    nn_predict = nn_model.predict(X_test)
    print(classification_report(y_test, nn_predict))
    print(confusion_matrix(y_test, nn_predict))

    print("** NN for PCA reduced data set")
    X_train_pca, _, _ = do_pca(X_train, None, pca_components)
    nn_model.fit(X_train_pca, y_train)
    X_test_pca, _, _ = do_pca(X_test, None, pca_components)
    nn_predict = nn_model.predict(X_test_pca)
    print(classification_report(y_test, nn_predict))
    print(confusion_matrix(y_test, nn_predict))

    print("** NN for ICA reduced data set")
    X_train_ica, _ = do_ica(X_train, ica_components)
    nn_model.fit(X_train_ica, y_train)
    X_test_ica, _ = do_ica(X_test, ica_components)
    nn_predict = nn_model.predict(X_test_ica)
    print(classification_report(y_test, nn_predict))
    print(confusion_matrix(y_test, nn_predict))

    print("** NN for Random Projection reduced data set")
    X_train_rand, _ = do_randomized_projections(X_train, rand_components)
    nn_model.fit(X_train_rand, y_train)
    X_test_rand, _ = do_randomized_projections(X_test, rand_components)
    nn_predict = nn_model.predict(X_test_rand)
    print(classification_report(y_test, nn_predict))
    print(confusion_matrix(y_test, nn_predict))

    print("** NN for Learn From Model reduced data set")
    X_train_lfm = do_learn_from_model(X_train, y_train, lfm_components)
    nn_model.fit(X_train_lfm, y_train)
    X_test_lfm = do_learn_from_model(X_test, y_test, lfm_components)  # hmm to make the test values, need the answers
    nn_predict = nn_model.predict(X_test_lfm)
    print(classification_report(y_test, nn_predict))
    print(confusion_matrix(y_test, nn_predict))

    # TODO Need to make results of kmeans and EM available as X-train data and run NN for both


def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(data, estimator[-1].labels_, metric="euclidean", sample_size=50000)
    ]

    # Show the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}"
                        "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))


def bench_em(gm, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    gm : GMM instance
        A :class:`~sklearn.mixture.GaussianMixture` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    if name != 'ica' and name != 'rnd-proj':
        estimator = make_pipeline(StandardScaler(), gm).fit(data)
    else:
        estimator = make_pipeline(None, gm).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].bic(data), estimator[-1].aic(data)]

    # Define the metrics which require only the true labels and estimator
    # labels
    pred = gm.predict(data)

    clustering_metrics = [
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, pred) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    if len(np.unique(pred)) > 1:  # if it only predicted 1 label silhouette will fail
        results += [
            metrics.silhouette_score(data, pred, sample_size=50000, random_state=1)
        ]
    else:
        results += [
            0.000
        ]
    # Show the results
    formatter_result = "{:9s}\t{:.3f}s\t{:17.1f}\t{:17.1f}\t{:.3f}\t{:.3f}\t{:4.3f}"
    print(formatter_result.format(*results))


def setup_projections(X, X_train_lfm, y_train_lfm, pca_components, ica_compoonents, rand_components, lfm_components):
    """
    must call before using global projection transforms
    """
    global X_transform_pca, X_transform_ica, X_transform_rand, X_transform_lfm
    t0 = time()
    X_transform_pca, _, pca = do_pca(X, None, n_components=pca_components)
    project_time = time() - t0
    print('PCA took %5.3fs' % project_time)

    t0 = time()
    X_transform_ica, ica = do_ica(X, n_components=ica_compoonents)
    project_time = time() - t0
    print('ICA took %5.3fs' % project_time)

    t0 = time()
    X_transform_rand, rand_proj = do_randomized_projections(X, n_components=rand_components)
    project_time = time() - t0
    print('rand took %5.3fs' % project_time)

    t0 = time()
    # learn from training answers not full data set, or it's cheating
    X_transform_lfm = do_learn_from_model(X_train_lfm, y_train_lfm, n_components=lfm_components)
    project_time = time() - t0
    print('lfm took %5.3fs' % project_time)


def do_k_means(X, y, y_train_lfm, n_classes):
    print(90 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tsilhouette')

    kmeans = KMeans(init="k-means++", n_clusters=n_classes, random_state=0)
    bench_k_means(kmeans=kmeans, name="k-means++", data=X, labels=y)

    kmeans = KMeans(init="random", n_clusters=n_classes, random_state=0)
    bench_k_means(kmeans=kmeans, name="random", data=X, labels=y)

    kmeans = KMeans(init="k-means++", n_clusters=n_classes, random_state=0)
    bench_k_means(kmeans=kmeans, name="PCA-based", data=X_transform_pca, labels=y)

    kmeans = KMeans(init="k-means++", n_clusters=n_classes, random_state=0)
    bench_k_means(kmeans=kmeans, name="ICA-based", data=X_transform_ica, labels=y)

    kmeans = KMeans(init="k-means++", n_clusters=n_classes, random_state=0)
    bench_k_means(kmeans=kmeans, name="rnd-proj", data=X_transform_rand, labels=y)

    kmeans = KMeans(init="k-means++", n_clusters=n_classes, random_state=0)
    bench_k_means(kmeans=kmeans, name="lfm", data=X_transform_lfm, labels=y_train_lfm)


def do_em(X, y, y_train_lfm, n_classes):
    """
    must exec do_k_means() first to fill global transform results
    """
    print(90 * '_')
    print('init\t\ttime\tbic\t\t\t\t\taic\t\t\t\t\tARI\t\tAMI\t\tsilhouette')
    if len(X[0:]) < 5000:  # don't do verbose output for faces data, not needed and hits div by zero bug (nice!)
        gm = GaussianMixture(n_components=n_classes, random_state=1)
    else:
        gm = GaussianMixture(n_components=n_classes, random_state=1, warm_start=True, verbose=2, verbose_interval=10)
    if len(X[0:]) < 5000:  # this takes too long to do without dimensions reduced
        bench_em(gm, name="base EM", data=X, labels=y)

    bench_em(gm, name="pca", data=X_transform_pca, labels=y)

    bench_em(gm, name="ica", data=X_transform_ica, labels=y)

    bench_em(gm, name="rnd-proj", data=X_transform_rand, labels=y)

    bench_em(gm, name="lfm", data=X_transform_lfm, labels=y_train_lfm)


def do_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=True, random_state=1).fit(X_train)
    X_train_transform = pca.transform(X_train)
    if X_test is None:
        return X_train_transform, None, pca
    X_test_transform = pca.transform(X_test)
    return X_train_transform, X_test_transform, pca


def do_ica(X, n_components):
    ica = FastICA(n_components=n_components, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None,
                  max_iter=200,
                  tol=0.001, w_init=None, random_state=1)
    X_transform = ica.fit_transform(X)
    return X_transform, ica


def do_randomized_projections(X, n_components):
    rand_proj = random_projection.GaussianRandomProjection(n_components=n_components, random_state=1)
    X_transform = rand_proj.fit_transform(X)
    return X_transform, rand_proj


def do_learn_from_model(X, y, n_components):
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=150, random_state=1, n_jobs=-1),
                                          max_features=n_components)
    embeded_rf_selector.fit(X, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    X_transform = X[:, embeded_rf_support]
    return X_transform


def do_the_grid(override_data_set):
    if override_data_set == None:
        data_set = ['faces', 'forest']  # have the more interesting data set last so projected data sets for it for NN
    else:
        data_set = [override_data_set]
    for curr_data_set in data_set:
        print('working on data set: ', curr_data_set)
        target_names, n_classes, X_train, X_test, y_train, y_test, X_orig, y_orig, X_train_lfm, y_train_lfm = \
            get_the_data(curr_data_set)

        pca_components = n_classes
        ica_compoonents = n_classes
        rand_components = n_classes
        lfm_components = n_classes

        setup_projections(X_orig, X_train_lfm, y_train_lfm,
                          pca_components, ica_compoonents, rand_components, lfm_components)

        do_k_means(X_orig, y_orig, y_train_lfm, n_classes)
        do_em(X_orig, y_orig, y_train_lfm, n_classes)

        do_nn(curr_data_set, X_train, y_train, X_test, y_test,
              pca_components, ica_compoonents, rand_components, lfm_components)


def get_the_data(dataset):
    if dataset == 'faces':
        # init Labeled Faces in the Wild dataset
        data_set = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    if dataset == 'forest':
        # init Forest Cover Types dataset
        data_set = fetch_covtype()
    if dataset == 'faces':
        _, h, w = data_set.images.shape
    # counts = np.bincount(data_set.target)
    # for i, (count, name) in enumerate(zip(counts, data_set.target_names)):
    #     print('{0:25}   {1:3}'.format(name, count))
    X_orig = data_set.data
    y_orig = data_set.target
    # Both cov_type & LFW are not balanced sets, at all.
    # So generate synthetic samples to make up minority class deficiencies
    if dataset == 'faces':
        oversample = SMOTE()
        X, y = oversample.fit_resample(X_orig, y_orig)
    if dataset == 'forest':
        file = pathlib.Path("forest_X_smoted.npy")
        if file.exists():
            X = load('forest_X_smoted.npy')
            y = load('forest_y_smoted.npy')
        else:
            oversample = SMOTE()
            X, y = oversample.fit_resample(X_orig, y_orig)
            save('forest_X_smoted.npy', X)
            save('forest_y_smoted', y)
    if dataset == 'faces':
        target_names = data_set.target_names
        n_classes = target_names.shape[0]
    if dataset == 'forest':
        target_names = np.unique(data_set.target)
        n_classes = len(target_names)
    if dataset == 'faces':
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        X_train_lfm, X_test_lfm, y_train_lfm, y_test_lfm = train_test_split(X, y, train_size=0.5, test_size=0.5,
                                                                            random_state=1)
    if dataset == 'forest':
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=10000, random_state=1)
        X_train_lfm, X_test_lfm, y_train_lfm, y_test_lfm = train_test_split(X, y, train_size=0.5, test_size=0.5,
                                                                            random_state=1)
    return target_names, n_classes, X_train, X_test, y_train, y_test, X_orig, y_orig, X_train_lfm, y_train_lfm


class Log:
    logging.basicConfig(filename='/tmp/numpy_warnings.log', encoding='utf-8',
                        format='%(asctime)s %(message)s', level=logging.DEBUG)

    def write(self, msg):
        logging.debug("%s" % msg, stack_info=True)


if __name__ == '__main__':
    log = Log()
    saved_handler = np.seterrcall(log)
    np.seterr(over='log')
    logging.info("starting run")
    # request data set as "faces" or "forest", in none passed, do both
    if len(sys.argv) > 1:
        data_set = sys.argv[1]
    else:
        data_set = None

    do_the_grid(data_set)

    DO_LEARNING_CURVES = False

    # if DO_LEARNING_CURVES:
    #     fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    #     title = r"Learning Curves (" + data_set + "_" + ALG + ")"
    #     cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    #     # plot_learning_curve(ada_model, title, X_train_transform, y_train, axes=axes[:, 0], ylim=(0.5, 1.01), cv=cv, n_jobs=-1)
    #     plt.show()
