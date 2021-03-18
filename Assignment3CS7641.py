# Assignment 3 Clustering and Dimensionality Reduction

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from dtreeviz.trees import *
from numpy import save
from numpy import load
import pathlib
from sklearn.neural_network import MLPClassifier


def grid_search_cv_nn_model(X_train, y_train):
    mlp_gs = MLPClassifier(max_iter=500)
    parameter_space = {
        'hidden_layer_sizes': [(10, 30, 10), (20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    model = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    model.fit(X_train, y_train)
    return model


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


def do_nn(dataset):
    if dataset == 'faces':
        nn_model = MLPClassifier(alpha=0.05, hidden_layer_sizes=(20,), learning_rate='adaptive', max_iter=500)
    if dataset == 'forest':
        nn_model = MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(20,),
                                 learning_rate='adaptive', max_iter=500)
    nn_model.fit(X_train_transform, y_train)
    nn_predict = nn_model.predict(X_test_transform)
    print(classification_report(y_test, nn_predict))
    print(confusion_matrix(y_test, nn_predict))


def do_k_means():
    pass


def do_em():
    pass


def do_pca(X_train_transform, X_test_transform):
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train_transform)
    X_train_transform = pca.transform(X_train_transform)
    X_test_transform = pca.transform(X_test_transform)
    return X_train_transform, X_test_transform

def do_ica():
    pass


def do_randomized_projections():
    pass


def do_some_other_feat_selection_algorithm():
    pass


def do_the_grid():
    pass


def get_the_data(dataset):
    if dataset == 'faces':
        # init Labeled Faces in the Wild dataset
        data_set = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    if dataset == 'forest':
        # init Forest Cover Types dataset
        data_set = fetch_covtype()
    if dataset == 'faces':
        _, h, w = data_set.images.shape

        counts = np.bincount(data_set.target)
        for i, (count, name) in enumerate(zip(counts, data_set.target_names)):
            print('{0:25}   {1:3}'.format(name, count))
    X = data_set.data
    y = data_set.target
    # Both cov_type & LFW are not balanced sets, at all.
    # So generate synthetic samples to make up minority class deficiencies
    if dataset == 'faces':
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
    if dataset == 'forest':
        counts = np.bincount(y)[1:]

        file = pathlib.Path("forest_X_smoted.npy")
        if file.exists():
            X = load('forest_X_smoted.npy')
            y = load('forest_y_smoted.npy')
        else:
            oversample = SMOTE()
            X, y = oversample.fit_resample(X, y)
            save('forest_X_smoted.npy', X)
            save('forest_y_smoted', y)
    if dataset == 'faces':
        n_features = X.shape[1]
        target_names = data_set.target_names
        n_classes = target_names.shape[0]
    if dataset == 'forest':
        n_features = X.shape[1]
        target_names = data_set.target_names
        n_classes = len(target_names)
    if dataset == 'faces':
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    if dataset == 'forest':
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=10000, random_state=1)
    X_train_transform = np.zeros_like(X_train)
    X_test_transform = np.zeros_like(X_test)
    if dataset == 'faces':
        n_components = 60
    if dataset == 'forest':
        n_components = 50
    return target_names, n_classes, X_train, X_test, y_train, y_test, X_train_transform, X_test_transform, n_components


if __name__ == '__main__':
    # request data set as "faces" or "forest"
    data_set = "forest"
    target_names, n_classes, X_train, X_test, y_train, y_test, X_train_transform, X_test_transform, n_components = \
        get_the_data(data_set)

    XFORM = 'PCA'
    # XFORM = 'None'

    ALG = 'NN'

    DO_LEARNING_CURVES = False

    if XFORM.__contains__('None'):
        # do no transforms
        X_train_transform = X_train
        X_test_transform = X_test
    if XFORM.__contains__('PCA'):
        X_train_transform, X_test_transform = do_pca(X_train, X_test)

    # TODO
    do_the_grid()

    if ALG.__contains__('NN'):
        do_nn(data_set)

    if DO_LEARNING_CURVES:
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        title = r"Learning Curves (" + DATA_SET + "_" + ALG + ")"
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # plot_learning_curve(ada_model, title, X_train_transform, y_train, axes=axes[:, 0], ylim=(0.5, 1.01), cv=cv, n_jobs=-1)
        plt.show()
