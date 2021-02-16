import errno

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from subprocess import call
from dtreeviz.trees import *
import re
import imblearn


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
    # just for dev speed, TODO: delete
    # tree_param = {'criterion': ['entropy'],
    #               'max_depth': [None],
    #               'min_samples_split': [3],
    #               'min_samples_leaf': [6],
    #               'max_features': [11]
    #               }
    model = GridSearchCV(dt_model, tree_param, cv=5, n_jobs=-1, error_score=0.0)
    model.fit(X_train, y_train)
    return model


def grid_search_cv_svm_model(X_train, y_train):
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    model = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=5, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


if __name__ == '__main__':
    # init Labeled Faces in the Wild dataset
    # reference:
    # Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.
    # Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. University of Massachusetts, Amherst,
    # Technical Report 07-49, October, 2007.
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    n_samples, h, w = lfw_people.images.shape

    X_lfw_people = lfw_people.data
    y_lfw_people = lfw_people.target

    # This is not a balanced set, at all so generate synthetic samples to make up class deficiencies
    oversample = SMOTE()
    X_lfw_people, y_lfw_people = oversample.fit_resample(X_lfw_people, y_lfw_people)

    n_features = X_lfw_people.shape[1]
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    X_train_lfw_people, X_test_lfw_people, y_train_lfw_people, y_test_lfw_people = \
        train_test_split(X_lfw_people, y_lfw_people, random_state=1)

    X_train_transform_lfw_people = np.zeros_like(X_train_lfw_people)
    X_test_transform_lfw_people = np.zeros_like(X_test_lfw_people)

    n_components = 60

    XFORM = 'JUST_PCA'
    # ALG = 'DTREE'
    # ALG = 'ADABOOST'
    ALG = 'SVM'

    if XFORM.__contains__('None'):
        # do no transforms
        X_train_transform_lfw_people = X_train_lfw_people
        X_test_transform_lfw_people = X_test_lfw_people
    if XFORM.__contains__('SS'):
        ss = StandardScaler()
        X_train_transform_lfw_people = ss.fit_transform(X_train_lfw_people)
        X_test_transform_lfw_people = ss.fit_transform(X_test_lfw_people)
    if XFORM.__contains__('LBP'):
        # try LBP manually
        from skimage.feature import local_binary_pattern

        for i, image in enumerate(X_train_lfw_people):
            X_train_transform_lfw_people[i] = np.ravel(local_binary_pattern(np.reshape(image, (h, w)), 24, 3))
        for i, image in enumerate(X_test_lfw_people):
            X_test_transform_lfw_people[i] = np.ravel(local_binary_pattern(np.reshape(image, (h, w)), 24, 3))
    if XFORM.__contains__('PCA'):
        pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train_transform_lfw_people)
        X_train_transform_lfw_people = pca.transform(X_train_transform_lfw_people)
        X_test_transform_lfw_people = pca.transform(X_test_transform_lfw_people)
    if XFORM.__contains__('JUST_PCA'):
        pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train_lfw_people)
        X_train_transform_lfw_people = pca.transform(X_train_lfw_people)
        X_test_transform_lfw_people = pca.transform(X_test_lfw_people)

    # init Forest Cover Types dataset
    # reference:
    # Blackard, Jock A. and Denis J. Dean. 2000. "Comparative Accuracies of Artificial Neural Networks and
    # Discriminant Analysis in Predicting Forest Cover Types from Cartographic Variables." Computers and
    # Electronics in Agriculture 24(3):131-151.
    cov_type = fetch_covtype()

    # FIXME cross_validate won't work as-is:
    # cv_results = cross_validate(ada_boost_scratch, X, y, cv=3)

    if ALG == 'ADABOOST':
        # adaboost on a good Decision Tree Set found via grid search
        ada_model = AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=11, max_features=18), n_estimators=500)
        ada_model.fit(X_train_transform_lfw_people, y_train_lfw_people)
        ada_predict = ada_model.predict(X_test_transform_lfw_people)
        print(classification_report(y_test_lfw_people, ada_predict, target_names=target_names))
        print(confusion_matrix(y_test_lfw_people, ada_predict, labels=range(n_classes)))
        exit(0)

    if ALG == 'SVM':
        svm_model = grid_search_cv_svm_model(X_train_transform_lfw_people, y_train_lfw_people)
        svm_predict = svm_model.predict(X_test_transform_lfw_people)
        print("Best SVM estimator found by grid search:")
        print(svm_model.best_estimator_)
        print(classification_report(y_test_lfw_people, svm_predict, target_names=target_names))
        print(confusion_matrix(y_test_lfw_people, svm_predict, labels=range(n_classes)))

    # Decision trees plain with grid search on parameters and cross-validation
    dte_model = grid_search_cv_decision_tree_model(X_train_transform_lfw_people, y_train_lfw_people)

    print("Best DTree estimator found by grid search:")
    print(dte_model.best_estimator_)
    dte_predict = dte_model.predict(X_test_transform_lfw_people)
    print(classification_report(y_test_lfw_people, dte_predict, target_names=target_names))
    print(confusion_matrix(y_test_lfw_people, dte_predict, labels=range(n_classes)))

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
                   x_data=X_train_transform_lfw_people,
                   y_data=y_train_lfw_people,
                   target_name='class',
                   feature_names=X_lfw_people[1],
                   class_names=list(target_names),
                   title="Decision Tree - Labeled Faces in the Wild data set")
    svg_filename = viz.save_svg()

    curr_viz_file_path = Path(svg_filename).rename(new_dtree_viz_path)
    print("dtreeviz svg file saved as: ", curr_viz_file_path)
