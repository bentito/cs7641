import mlrose_hiive as mlrose
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA


def knapsack_setup(max_weight_pct):
    weights = [10, 5, 2, 8, 15]
    values = [1, 2, 3, 4, 5]
    fitness = mlrose.Knapsack(weights, values, max_weight_pct, max_item_count=100)
    return mlrose.DiscreteOpt(length=5, fitness_fn=fitness, max_val=100)


def six_peaks_setup(t_pct):
    fitness = mlrose.SixPeaks(t_pct)
    return mlrose.DiscreteOpt(length=12, fitness_fn=fitness, max_val=2)


def flip_flop_setup(t_pct):
    fitness = mlrose.FlipFlop()
    return mlrose.DiscreteOpt(length=12, fitness_fn=fitness, max_val=2)


def do_knapsack_analysis(problem, schedule, init_state):
    best_state = np.zeros((4, len(init_state)))
    best_fit = np.zeros(4)
    fitness_curve = np.zeros((4, 100, 2))
    best_state[0], best_fit[0], fitness_curve[0] = mlrose.simulated_annealing(problem,
                                                                              schedule=schedule,
                                                                              max_attempts=1000,
                                                                              max_iters=1000,
                                                                              init_state=init_state,
                                                                              curve=False,
                                                                              random_state=1)
    best_state[1], best_fit[1], fitness_curve[1] = mlrose.genetic_alg(problem,
                                                                      max_attempts=1000,
                                                                      max_iters=1000,
                                                                      curve=False,
                                                                      random_state=1)
    best_state[2], best_fit[2], fitness_curve[2] = mlrose.random_hill_climb(problem,
                                                                            restarts=1000,
                                                                            max_iters=1000,
                                                                            init_state=init_state,
                                                                            curve=False,
                                                                            random_state=1)
    best_state[3], best_fit[3], fitness_curve[3] = mlrose.mimic(problem,
                                                                max_attempts=1000,
                                                                max_iters=1000,
                                                                curve=False,
                                                                random_state=1)
    return best_state, best_fit, fitness_curve


if __name__ == '__main__':
    NN = True
    # fitness = mlrose.Queens()
    # problem = mlrose.DiscreteOpt(length=8, fitness_fn=fitness, maximize=False, max_val=8)
    # schedule = mlrose.ExpDecay()
    # init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    #
    # best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(problem,
    #                                                                      schedule=schedule,
    #                                                                      max_attempts=100,
    #                                                                      max_iters=1000,
    #                                                                      init_state=init_state,
    #                                                                      random_state=1)
    # print(best_state)
    # print(best_fitness)
    if not NN:
        max_weight_pct = 0.6
        problem = knapsack_setup(max_weight_pct)
        schedule = mlrose.ExpDecay()
        init_state = np.array([1, 0, 2, 1, 0])
        best_state, best_fitness, fitness_curve = do_knapsack_analysis(problem, schedule, init_state)
        best_idx = np.argmax(best_fitness)
        if best_idx == 0:
            alg = "SA"
        if best_idx == 1:
            alg = "GA"
        if best_idx == 2:
            alg = "RHC"
        if best_idx == 3:
            alg = "MIMIC"
        print("knapsack_best is %s with best state & best fitness: %s, %f" % (
            alg, np.array2string(best_state[best_idx]), best_fitness[best_idx]))
    else:
        data_set = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        X = data_set.data
        y = data_set.target
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        n_features = X.shape[1]
        target_names = data_set.target_names
        n_classes = target_names.shape[0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        n_components = 60
        pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train)
        X_train_transform = pca.transform(X_train)
        X_test_transform = pca.transform(X_test)
        one_hot = OneHotEncoder()
        y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
        y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

        # Initialize neural network object and fit object
        nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[2],
                                         activation='relu',
                                         algorithm='random_hill_climb',
                                         max_iters=1000,
                                         bias=True,
                                         is_classifier=True,
                                         learning_rate=0.0001,
                                         early_stopping=True,
                                         clip_max=5,
                                         max_attempts=100,
                                         random_state=3)

        nn_model1.fit(X_train_transform, y_train_hot)
        y_train_pred = nn_model1.predict(X_train_transform)
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
        print(y_train_accuracy)
        y_test_pred = nn_model1.predict(X_test_transform)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
        print(y_test_accuracy)
