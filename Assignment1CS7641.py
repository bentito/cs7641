import errno
import time

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split, learning_curve, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from dtreeviz.trees import *
import re
from numpy import save
from numpy import load
import pathlib
from sklearn.neural_network import MLPClassifier


def grid_search_cv_decision_tree_model(X_train, y_train) -> DecisionTreeClassifier:
    dt_model = DecisionTreeClassifier()
    tree_param = {'criterion': ('gini', 'entropy'),
                  'max_depth': np.arange(1, 12),
                  'min_samples_split': np.arange(2, 7),
                  'min_samples_leaf': np.arange(1, 7),
                  'max_features': np.arange(1, 20)
                  }
    model = GridSearchCV(dt_model, tree_param, cv=5, n_jobs=-1, error_score=0.0)
    model.fit(X_train, y_train)
    return model


def grid_search_cv_svm_model(X_train, y_train, DATA_SET):
    if DATA_SET == 'faces':
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        model = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=5, n_jobs=-1)
    if DATA_SET == 'forest':
        param_grid = {'kernel': ['linear']}
        model = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


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


def grid_search_cv_knn_model(X_train, y_train):
    pipe = Pipeline([
        ('sc', StandardScaler()),
        ('knn', KNeighborsClassifier(algorithm='auto'))
    ])
    params = {
        'knn__n_neighbors': [1, 3, 5, 7, 9, 11],
        'knn__weights': ['uniform', 'distance']
    }
    model = GridSearchCV(pipe, param_grid=params, n_jobs=-1, cv=5)
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


if __name__ == '__main__':
    DATA_SET = "forest"
    # DATA_SET = "faces"

    if DATA_SET == 'faces':
        # init Labeled Faces in the Wild dataset
        data_set = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    if DATA_SET == 'forest':
        # init Forest Cover Types dataset
        data_set = fetch_covtype()

    if DATA_SET == 'faces':
        _, h, w = data_set.images.shape

    X = data_set.data
    y = data_set.target

    # Both cov_type & LFW are not balanced sets, at all.
    # So generate synthetic samples to make up minority class deficiencies
    if DATA_SET == 'faces':
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
    if DATA_SET == 'forest':
        file = pathlib.Path("forest_X_smoted.npy")
        if file.exists():
            X = load('forest_X_smoted.npy')
            y = load('forest_y_smoted.npy')
        else:
            oversample = SMOTE()
            X, y = oversample.fit_resample(X, y)
            save('forest_X_smoted.npy', X)
            save('forest_y_smoted', y)

    if DATA_SET == 'faces':
        n_features = X.shape[1]
        target_names = data_set.target_names
        n_classes = target_names.shape[0]
    if DATA_SET == 'forest':
        n_features = X.shape[1]
        target_names = data_set.target_names
        n_classes = len(target_names)

    if DATA_SET == 'faces':
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    if DATA_SET == 'forest':
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=10000, random_state=1)

    X_train_transform = np.zeros_like(X_train)
    X_test_transform = np.zeros_like(X_test)

    if DATA_SET == 'faces':
        n_components = 60
    if DATA_SET == 'forest':
        n_components = 50

    XFORM = 'JUST_P_CA'
    # XFORM = 'SS+PCA'
    # XFORM = 'SS'
    # XFORM = 'None'
    # XFORM = 'MINMAX'

    ALG = 'DTREE'
    # ALG = 'ADABOOST'
    # ALG = 'SVM'
    # ALG = 'NN'
    # ALG = 'K_NN'

    DO_LEARNING_CURVES = True

    if XFORM.__contains__('None'):
        # do no transforms
        X_train_transform = X_train
        X_test_transform = X_test
    if XFORM.__contains__('SS'):
        ss = StandardScaler()
        X_train_transform = ss.fit_transform(X_train)
        X_test_transform = ss.fit_transform(X_test)
    if XFORM.__contains__('LBP'):
        # try LBP manually
        from skimage.feature import local_binary_pattern

        for i, image in enumerate(X_train):
            X_train_transform[i] = np.ravel(local_binary_pattern(np.reshape(image, (h, w)), 24, 3))
        for i, image in enumerate(X_test):
            X_test_transform[i] = np.ravel(local_binary_pattern(np.reshape(image, (h, w)), 24, 3))
    if XFORM.__contains__('PCA'):
        pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train_transform)
        X_train_transform = pca.transform(X_train_transform)
        X_test_transform = pca.transform(X_test_transform)
    if XFORM.__contains__('JUST_P_CA'):
        pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train)
        X_train_transform = pca.transform(X_train)
        X_test_transform = pca.transform(X_test)
    if XFORM.__contains__('MINMAX'):
        mm = make_pipeline(MinMaxScaler(), Normalizer())
        X_train_transform = mm.fit_transform(X_train)
        X_test_transform = mm.transform(X_test)

    if ALG.__contains__('ADABOOST'):
        # adaboost on a good Decision Tree Set found via grid search
        if DATA_SET == 'faces':
            ada_model = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=11, max_features=18),
                n_estimators=500)
        if DATA_SET == 'forest':
            ada_model = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=11, max_features=19,
                                                      min_samples_split=5), n_estimators=500)
        ada_model.fit(X_train_transform, y_train)
        ada_predict = ada_model.predict(X_test_transform)
        if DATA_SET == 'faces':
            print(classification_report(y_test, ada_predict, target_names=target_names))
            print(confusion_matrix(y_test, ada_predict, labels=range(n_classes)))
        if DATA_SET == 'forest':
            print(classification_report(y_test, ada_predict))
            print(confusion_matrix(y_test, ada_predict))

    if ALG.__contains__('SVM'):
        svm_model = grid_search_cv_svm_model(X_train_transform, y_train, DATA_SET)
        svm_predict = svm_model.predict(X_test_transform)
        print("Best SVM estimator found by grid search:")
        print(svm_model.best_estimator_)
        if DATA_SET == 'faces':
            print(classification_report(y_test, svm_predict, target_names=target_names))
            print(confusion_matrix(y_test, svm_predict, labels=range(n_classes)))
        if DATA_SET == 'forest':
            print(classification_report(y_test, svm_predict))
            print(confusion_matrix(y_test, svm_predict))

    if ALG.__contains__('NN'):
        nn_model = grid_search_cv_nn_model(X_train_transform, y_train)
        nn_predict = nn_model.predict(X_test_transform)
        print("Best NN estimator found by grid search:")
        print(nn_model.best_estimator_)
        if DATA_SET == 'faces':
            print(classification_report(y_test, nn_predict, target_names=target_names))
            print(confusion_matrix(y_test, nn_predict, labels=range(n_classes)))
        if DATA_SET == 'forest':
            print(classification_report(y_test, nn_predict))
            print(confusion_matrix(y_test, nn_predict))

    if ALG.__contains__('K_NN'):
        knn_model = grid_search_cv_knn_model(X_train_transform, y_train)
        knn_predict = knn_model.predict(X_test_transform)
        print("Best KNN estimator found by grid search:")
        print(knn_model.best_estimator_)
        if DATA_SET == 'faces':
            print(classification_report(y_test, knn_predict, target_names=target_names))
            print(confusion_matrix(y_test, knn_predict, labels=range(n_classes)))
        if DATA_SET == 'forest':
            print(classification_report(y_test, knn_predict))
            print(confusion_matrix(y_test, knn_predict))

    if ALG.__contains__('DTREE'):
        # Decision trees plain with grid search on parameters and cross-validation
        dte_model = grid_search_cv_decision_tree_model(X_train_transform, y_train)
        print("Best DTree estimator found by grid search:")
        print(dte_model.best_estimator_)
        dte_predict = dte_model.predict(X_test_transform)
        if DATA_SET == 'faces':
            print(classification_report(y_test, dte_predict, target_names=target_names))
            print(confusion_matrix(y_test, dte_predict, labels=range(n_classes)))
        if DATA_SET == 'forest':
            print(classification_report(y_test, dte_predict))
            print(confusion_matrix(y_test, dte_predict))

        # make visualization for dtrees
        DTREE_VIZ_DIR = 'dtree_viz_images'
        try:
            os.makedirs(DTREE_VIZ_DIR)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        new_dtree_viz_path = os.path.join(DTREE_VIZ_DIR, DATA_SET + "_" + XFORM + "_" +
                                          re.sub('\s+', '_', str(dte_model.best_estimator_).
                                                 replace('(', '_').
                                                 replace(')', '_').
                                                 replace("'", '').
                                                 replace("\n", '').
                                                 replace(',', '')) +
                                          str(n_components)) + '.svg'
        if DATA_SET == 'faces':
            viz = dtreeviz(dte_model.best_estimator_,
                           x_data=X_train_transform,
                           y_data=y_train,
                           target_name='class',
                           feature_names=X[1],
                           class_names=list(target_names),
                           title="Decision Tree - Labeled Faces in the Wild data set")
        if DATA_SET == 'forest':
            viz = dtreeviz(dte_model.best_estimator_,
                           x_data=X_train_transform,
                           y_data=y_train,
                           target_name='class',
                           feature_names=X[1],
                           class_names=np.unique(y).tolist(),
                           title="Decision Tree - Forest Cover Types data set")
        svg_filename = viz.save_svg()

        curr_viz_file_path = Path(svg_filename).rename(new_dtree_viz_path)
        print("dtreeviz svg file saved as: ", curr_viz_file_path)

    if DO_LEARNING_CURVES:
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        title = r"Learning Curves ("+ALG+")"
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        plot_learning_curve(dte_model, title, X_train_transform, y_train, axes=axes[:, 0], ylim=(0.5, 1.01), cv=cv, n_jobs=-1)
        plt.show()
