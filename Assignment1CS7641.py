import errno
import time

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
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


# noinspection PyPep8Naming
def ada_boost_scratch(X, y, M=10, learning_rate=1):
    # Initialization of utility variables
    N = len(y)
    estimator_list, y_predict_list, estimator_error_list, estimator_weight_list, sample_weight_list = [], [], [], [], []

    # Initialize the sample weights
    sample_weight = np.ones(N) / N
    sample_weight_list.append(sample_weight.copy())

    # For m = 1 to M
    for m in range(M):
        # Fit a classifier
        estimator = DecisionTreeClassifier(max_depth=11, max_features=19, min_samples_split=3)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)

        # Misclassifications
        incorrect = (y_predict != y)

        # Estimator error
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Boost estimator weights
        estimator_weight = learning_rate * np.log((1. - estimator_error) / estimator_error)

        # Boost sample weights
        sample_weight *= np.exp(estimator_weight * incorrect * ((sample_weight > 0) | (estimator_weight < 0)))

        # Save iteration values
        estimator_list.append(estimator)
        y_predict_list.append(y_predict.copy())
        estimator_error_list.append(estimator_error.copy())
        estimator_weight_list.append(estimator_weight.copy())
        sample_weight_list.append(sample_weight.copy())

    # Convert to np array for convenience
    estimator_list = np.asarray(estimator_list)
    y_predict_list = np.asarray(y_predict_list)
    estimator_error_list = np.asarray(estimator_error_list)
    estimator_weight_list = np.asarray(estimator_weight_list)
    sample_weight_list = np.asarray(sample_weight_list)

    # Predictions
    predictions = (np.array([np.sign((y_predict_list[:, point] * estimator_weight_list).sum()) for point in range(N)]))
    print('Accuracy = ', (predictions == y).sum() / N)

    return estimator_list, estimator_weight_list, sample_weight_list


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
    mlp_gs = MLPClassifier(max_iter=250)
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

    # XFORM = 'JUST_P_CA'
    # XFORM = 'SS'
    # XFORM = 'None'
    XFORM = 'MINMAX'

    # ALG = 'DTREE'
    # ALG = 'ADABOOST'
    ALG = 'SVM'
    # ALG = 'NN'
    # ALG = 'KNN'

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

    if ALG == 'ADABOOST':
        # adaboost on a good Decision Tree Set found via grid search
        ada_model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=11, max_features=18), n_estimators=500)
        ada_model.fit(X_train_transform, y_train)
        ada_predict = ada_model.predict(X_test_transform)
        print(classification_report(y_test, ada_predict, target_names=target_names))
        print(confusion_matrix(y_test, ada_predict, labels=range(n_classes)))
        exit(0)

    if ALG == 'SVM':
        svm_model = grid_search_cv_svm_model(X_train_transform, y_train, DATA_SET)
        svm_predict = svm_model.predict(X_test_transform)
        print("Best SVM estimator found by grid search:")
        print(svm_model.best_estimator_)
        if DATA_SET == 'faces':
            print(classification_report(y_test, svm_predict, target_names=target_names))
        if DATA_SET == 'forest':
            print(classification_report(y_test, svm_predict))
        print(confusion_matrix(y_test, svm_predict))
        exit(0)

    if ALG == 'NN':
        nn_model = grid_search_cv_nn_model(X_train_transform, y_train)
        nn_predict = nn_model.predict(X_test_transform)
        print("Best NN estimator found by grid search:")
        print(nn_model.best_estimator_)
        print(classification_report(y_test, nn_predict, target_names=target_names))
        print(confusion_matrix(y_test, nn_predict, labels=range(n_classes)))
        exit(0)

    if ALG == 'KNN':
        knn_model = grid_search_cv_knn_model(X_train_transform, y_train)
        knn_predict = knn_model.predict(X_test_transform)
        print("Best KNN estimator found by grid search:")
        print(knn_model.best_estimator_)
        if DATA_SET == 'faces':
            print(classification_report(y_test, knn_predict, target_names=target_names))
        if DATA_SET == 'forest':
            print(classification_report(y_test, knn_predict, labels=range(n_classes)))
        print(confusion_matrix(y_test, knn_predict, labels=range(n_classes)))
        exit(0)

    # Decision trees plain with grid search on parameters and cross-validation
    dte_model = grid_search_cv_decision_tree_model(X_train_transform, y_train)

    print("Best DTree estimator found by grid search:")
    print(dte_model.best_estimator_)
    dte_predict = dte_model.predict(X_test_transform)
    if DATA_SET == 'faces':
        print(classification_report(y_test, dte_predict, target_names=target_names))
    if DATA_SET == 'forest':
        print(classification_report(y_test, dte_predict, labels=range(n_classes)))
    print(confusion_matrix(y_test, dte_predict, labels=range(n_classes)))

    # make visualization for dtrees
    DTREE_VIZ_DIR = 'dtree_viz_images'
    try:
        os.makedirs(DTREE_VIZ_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    new_dtree_viz_path = os.path.join(DTREE_VIZ_DIR, XFORM + re.sub('\s+', '_', str(dte_model.best_estimator_).
                                                                    replace('(', '_').
                                                                    replace(')', '_').
                                                                    replace("'", '').
                                                                    replace("\n", '').
                                                                    replace(',', '')) +
                                      str(n_components)) + '.svg'

    viz = dtreeviz(dte_model.best_estimator_,
                   x_data=X_train_transform,
                   y_data=y_train,
                   target_name='class',
                   feature_names=X[1],
                   class_names=list(target_names),
                   title="Decision Tree - Labeled Faces in the Wild data set")
    svg_filename = viz.save_svg()

    curr_viz_file_path = Path(svg_filename).rename(new_dtree_viz_path)
    print("dtreeviz svg file saved as: ", curr_viz_file_path)
