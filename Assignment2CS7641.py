import csv

import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import numpy as np
import time
from imblearn.over_sampling import SMOTE
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from textwrap import wrap


def knapsack_problem_setup(num_items, max_weight_pct, max_val):
    weights = list(np.random.randint(low=1, high=100, size=num_items))
    values = list(np.random.randint(low=1, high=100, size=num_items))
    fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    return mlrose.DiscreteOpt(length=num_items, fitness_fn=fitness, max_val=max_val)


def six_peaks_problem_setup(t_pct, num_items):
    fitness = mlrose.SixPeaks(t_pct)
    return mlrose.DiscreteOpt(length=num_items, fitness_fn=fitness, max_val=2)


def flip_flop_problem_setup():
    fitness = mlrose.FlipFlop()
    return mlrose.DiscreteOpt(length=12, fitness_fn=fitness, max_val=2)


def do_knapsack_analysis(problem, schedule, max_attempts, max_iters, keep_pct):
    best_state = np.zeros((4, problem.length))
    best_fit = np.zeros(4)
    times = np.zeros(4)
    fitness_curve = np.zeros((4, max_iters, 2))
    start = time.process_time_ns()
    best_state[0], best_fit[0], fit_curve = mlrose.simulated_annealing(problem,
                                                                       schedule=schedule,
                                                                       max_attempts=max_attempts,
                                                                       max_iters=max_iters,
                                                                       curve=True,
                                                                       random_state=1)
    end = time.process_time_ns()
    times[0] = end - start
    fitness_curve[0], fit_curve = pad_in_fit_curve(fitness_curve[0], fit_curve)

    start = time.process_time_ns()
    best_state[1], best_fit[1], fit_curve = mlrose.genetic_alg(problem,
                                                               max_attempts=max_attempts,
                                                               max_iters=max_iters,
                                                               curve=True,
                                                               random_state=1)
    end = time.process_time_ns()
    times[1] = end - start
    fitness_curve[1], fit_curve = pad_in_fit_curve(fitness_curve[1], fit_curve)

    start = time.process_time_ns()
    best_state[2], best_fit[2], fit_curve = mlrose.random_hill_climb(problem,
                                                                     restarts=100,
                                                                     max_attempts=max_attempts,
                                                                     max_iters=max_iters,
                                                                     curve=True,
                                                                     random_state=1)
    end = time.process_time_ns()
    times[2] = end - start
    fitness_curve[2], fit_curve = pad_in_fit_curve(fitness_curve[2], fit_curve)

    start = time.process_time_ns()
    best_state[3], best_fit[3], fit_curve = mlrose.mimic(problem,
                                                         max_attempts=max_attempts,
                                                         max_iters=max_iters,
                                                         keep_pct=keep_pct,
                                                         curve=True,
                                                         random_state=1)
    fitness_curve[3], fit_curve = pad_in_fit_curve(fitness_curve[3], fit_curve)
    end = time.process_time_ns()
    times[3] = end - start
    return best_state, best_fit, fitness_curve, times


def do_sixpeaks_analysis(problem, schedule, max_attempts, max_iters, keep_pct):
    best_state = np.zeros((4, problem.length))
    best_fit = np.zeros(4)
    times = np.zeros(4)
    fitness_curve = np.zeros((4, max_iters, 2))
    start = time.process_time_ns()
    best_state[0], best_fit[0], fit_curve = mlrose.simulated_annealing(problem,
                                                                       schedule=schedule,
                                                                       max_attempts=max_attempts,
                                                                       max_iters=max_iters,
                                                                       curve=True,
                                                                       random_state=1)
    end = time.process_time_ns()
    times[0] = end - start
    fitness_curve[0], fit_curve = pad_in_fit_curve(fitness_curve[0], fit_curve)

    start = time.process_time_ns()
    best_state[1], best_fit[1], fit_curve = mlrose.genetic_alg(problem,
                                                               max_attempts=max_attempts,
                                                               max_iters=max_iters,
                                                               curve=True,
                                                               random_state=1)
    end = time.process_time_ns()
    times[1] = end - start
    fitness_curve[1], fit_curve = pad_in_fit_curve(fitness_curve[1], fit_curve)

    start = time.process_time_ns()
    best_state[2], best_fit[2], fit_curve = mlrose.random_hill_climb(problem,
                                                                     restarts=100,
                                                                     max_attempts=max_attempts,
                                                                     max_iters=max_iters,
                                                                     curve=True,
                                                                     random_state=1)
    end = time.process_time_ns()
    times[2] = end - start
    fitness_curve[2], fit_curve = pad_in_fit_curve(fitness_curve[2], fit_curve)

    start = time.process_time_ns()
    best_state[3], best_fit[3], fit_curve = mlrose.mimic(problem,
                                                         max_attempts=max_attempts,
                                                         max_iters=max_iters,
                                                         keep_pct=keep_pct,
                                                         curve=True,
                                                         random_state=1)
    fitness_curve[3], fit_curve = pad_in_fit_curve(fitness_curve[3], fit_curve)
    end = time.process_time_ns()
    times[3] = end - start
    return best_state, best_fit, fitness_curve, times


def pad_in_fit_curve(larger_array, sm_array):
    larger_array[:sm_array.shape[0], :sm_array.shape[1]] = sm_array
    sm_array = []
    return larger_array, sm_array


def longest_fit_curve(fitness_curve):
    longest = np.max(np.nonzero(fitness_curve))
    return longest


def plot_fitness_curves(fitness_curve, num_items, annealing_schedule, max_weight_pct, max_iters, max_attempts, max_val,
                        keep_pct, times):
    times = times / 1000000
    longest_fit = longest_fit_curve(fitness_curve)
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.plot(range(longest_fit), fitness_curve[0, :longest_fit, 0], 'o-')
    plt.plot(range(longest_fit), fitness_curve[1, :longest_fit, 0], 'o-')
    plt.plot(range(longest_fit), fitness_curve[2, :longest_fit, 0], 'o-')
    plt.plot(range(longest_fit), fitness_curve[3, :longest_fit, 0], 'o-')
    plt.title("\n".join(wrap("Fitness Curves (num_items: " + str(num_items) +
                             ")(anneal_sched: " + annealing_schedule +
                             ")(max_weight_pct:" + str(max_weight_pct) +
                             ")(max_iters:" + str(max_iters) +
                             ")(max_val:" + str(max_val) +
                             ")(keep_pct:" + str(keep_pct) +
                             ")(max_attempts:" + str(max_attempts) + ")", 40)))
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend(["simulated annealing" + ' [time: ' + str(times[0]) + 'ms]',
                "genetic algorithm" + ' [time: ' + str(times[1]) + 'ms]',
                "hill climb" + ' [time: ' + str(times[2]) + 'ms]',
                "mimic" + ' [time: ' + str(times[3]) + 'ms]'],
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    NN = False
    GRID_SEARCH_KNAPSACK = False
    PLOT_SPECIALS_KNAPSACK = False
    RUN_SPECIALS_KNAPSACK = False
    GRID_SEARCH_SIX_PEAKS = False
    PLOT_SPECIALS_SIX_PEAKS = False
    RUN_SPECIALS_SIX_PEAKS = True
    if not NN:
        schedules = [mlrose.ExpDecay(), mlrose.GeomDecay(), mlrose.ArithDecay()]
        annealing_schedules = ['exp_decay', 'geom_decay', 'arith_decay']
        max_weight_pcts = [0.2, 0.35]
        num_itemses = [5, 7, 12]
        max_iterses = [200, 1000]
        max_attemptses = [5, 10, 15]
        max_vals = [2, 5]
        keep_pcts = [0.2, 0.4]
        # knapsack
        justOneHeader = True
        if GRID_SEARCH_KNAPSACK:
            with open('gridsearch_knapsack.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for keep_pct in keep_pcts:
                    for max_val in max_vals:
                        for max_iters in max_iterses:
                            for num_items in num_itemses:
                                for schedule, annealing_schedule in zip(schedules, annealing_schedules):
                                    for max_weight_pct in max_weight_pcts:
                                        problem = knapsack_problem_setup(num_items, max_weight_pct, max_val)
                                        for max_attempts in max_attemptses:
                                            best_state, best_fitness, fitness_curve, times = do_knapsack_analysis(
                                                problem, schedule,
                                                max_attempts,
                                                max_iters,
                                                keep_pct)
                                            best_idx = np.argmax(best_fitness)
                                            if best_idx == 0:
                                                alg = "SA"
                                            if best_idx == 1:
                                                alg = "GA"
                                            if best_idx == 2:
                                                alg = "RHC"
                                            if best_idx == 3:
                                                alg = "MIMIC"
                                            print("knapsack_best is %s " %
                                                  alg)
                                            print("    with best state & best fitness: %s, %f" %
                                                  (np.array2string(best_state[best_idx]), best_fitness[best_idx]))
                                            print(
                                                "         for max_att: %d, max_weight_pct: %f, schedule: %s, num_items: %d," %
                                                (max_attempts, max_weight_pct, annealing_schedule, num_items))
                                            print("             max_iters: %d, time: %f" %
                                                  (max_iters, times[best_idx] / 1000000))
                                            if justOneHeader:
                                                spamwriter.writerow(
                                                    ['alg', 'best_state', 'best_fitness', 'max_att', 'max_wgt',
                                                     'annealing', 'num_items', 'max_iters', 'max_val', 'keep_pct',
                                                     'time'])
                                                justOneHeader = False
                                            spamwriter.writerow(
                                                [alg, np.array2string(best_state[best_idx]), best_fitness[best_idx],
                                                 max_attempts, max_weight_pct, annealing_schedule, num_items, max_iters,
                                                 max_val, times[best_idx] / 1000000])
        if PLOT_SPECIALS_KNAPSACK:
            # num_items, max_weight_pct, max_val, schedule, annealing_schedule, max_attempts, max_iters = 7, 0.35, 5, mlrose.ExpDecay(), 'exp_decay', 5, 200
            # problem = knapsack_problem_setup(num_items, max_weight_pct, max_val)
            # best_state, best_fitness, fitness_curve, times = do_knapsack_analysis(problem, schedule, max_attempts, max_iters)
            # plot_fitness_curves(fitness_curve, num_items, annealing_schedule, max_weight_pct, max_iters, max_attempts, max_val, times)
            #
            num_items, max_weight_pct, max_val, schedule, annealing_schedule, max_attempts, max_iters, keep_pct = 7, 0.35, 5, mlrose.ExpDecay(), 'exp_decay', 5, 5000, 0.3
            problem = knapsack_problem_setup(num_items, max_weight_pct, max_val)
            best_state, best_fitness, fitness_curve, times = do_knapsack_analysis(problem, schedule, max_attempts,
                                                                                  max_iters, keep_pct)
            plot_fitness_curves(fitness_curve, num_items, annealing_schedule, max_weight_pct, max_iters, max_attempts,
                                max_val, keep_pct, times)
        if RUN_SPECIALS_KNAPSACK:
            num_items, max_weight_pct, max_val, schedule, annealing_schedule, max_attempts, max_iters, keep_pct = 7, 0.35, 5, mlrose.ExpDecay(), 'exp_decay', 5, 5000, 0.3
            best_fitnesses = []
            NUM_RUNS = 100
            best_fitnesses = np.zeros((NUM_RUNS, 4))
            best_timeses = np.zeros((NUM_RUNS, 4))
            for idx, runs in enumerate(range(NUM_RUNS)):
                problem = knapsack_problem_setup(num_items, max_weight_pct, max_val)
                best_state, best_fitness, fitness_curve, times = do_knapsack_analysis(problem, schedule, max_attempts,
                                                                                      max_iters, keep_pct)
                best_fitnesses[idx] = best_fitness
                best_timeses[idx] = times
            print("mean knapsack fitnesses: " + str(np.mean(best_fitnesses, axis=0)))
            print("time weighted fitnesses: " + str(np.average(best_fitnesses, axis=0, weights=1. / best_timeses)))
        if GRID_SEARCH_SIX_PEAKS:
            num_itemses = [20, 60, 80, 100]
            max_vals = [2]
            with open('gridsearch_sixpeaks.csv', 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for keep_pct in keep_pcts:
                    for max_val in max_vals:
                        for max_iters in max_iterses:
                            for num_items in num_itemses:
                                for schedule, annealing_schedule in zip(schedules, annealing_schedules):
                                    for max_weight_pct in max_weight_pcts:
                                        problem = six_peaks_problem_setup(0.1, num_items)
                                        for max_attempts in max_attemptses:
                                            best_state, best_fitness, fitness_curve, times = do_sixpeaks_analysis(
                                                problem, schedule,
                                                max_attempts,
                                                max_iters,
                                                keep_pct)
                                            best_idx = np.argmax(best_fitness)
                                            if best_idx == 0:
                                                alg = "SA"
                                            if best_idx == 1:
                                                alg = "GA"
                                            if best_idx == 2:
                                                alg = "RHC"
                                            if best_idx == 3:
                                                alg = "MIMIC"
                                            print("sixpeaks_best is %s " %
                                                  alg)
                                            print("    with best state & best fitness: %s, %f" %
                                                  (np.array2string(best_state[best_idx]), best_fitness[best_idx]))
                                            print(
                                                "         for max_att: %d, max_weight_pct: %f, schedule: %s, num_items: %d," %
                                                (max_attempts, max_weight_pct, annealing_schedule, num_items))
                                            print("             max_iters: %d, time: %f" %
                                                  (max_iters, times[best_idx] / 1000000))
                                            if justOneHeader:
                                                spamwriter.writerow(
                                                    ['alg', 'best_state', 'best_fitness', 'max_att', 'max_wgt',
                                                     'annealing', 'num_items', 'max_iters', 'max_val', 'keep_pct',
                                                     'time'])
                                                justOneHeader = False
                                            spamwriter.writerow(
                                                [alg, np.array2string(best_state[best_idx]), best_fitness[best_idx],
                                                 max_attempts, max_weight_pct, annealing_schedule, num_items, max_iters,
                                                 max_val, times[best_idx] / 1000000])
        if PLOT_SPECIALS_SIX_PEAKS:
            num_items, max_weight_pct, max_val, schedule, annealing_schedule, max_attempts, max_iters, keep_pct = 80, 0.35, 2, mlrose.ExpDecay(), 'exp_decay', 5, 5000, 0.3
            problem = six_peaks_problem_setup(0.1, num_items)
            best_state, best_fitness, fitness_curve, times = do_sixpeaks_analysis(problem, schedule, max_attempts,
                                                                                  max_iters, keep_pct)
            plot_fitness_curves(fitness_curve, num_items, annealing_schedule, max_weight_pct, max_iters, max_attempts,
                                max_val, keep_pct, times)
        if RUN_SPECIALS_SIX_PEAKS:
            num_items, max_weight_pct, max_val, schedule, annealing_schedule, max_attempts, max_iters, keep_pct = 80, 0.35, 2, mlrose.ExpDecay(), 'exp_decay', 5, 5000, 0.3
            best_fitnesses = []
            NUM_RUNS = 3
            best_fitnesses = np.zeros((NUM_RUNS, 4))
            best_timeses = np.zeros((NUM_RUNS, 4))
            for idx, runs in enumerate(range(NUM_RUNS)):
                problem = six_peaks_problem_setup(0.1, num_items)
                best_state, best_fitness, fitness_curve, times = do_sixpeaks_analysis(problem, schedule, max_attempts,
                                                                                      max_iters, keep_pct)
                best_fitnesses[idx] = best_fitness
                best_timeses[idx] = times
            print("mean sixpeak fitnesses: " + str(np.mean(best_fitnesses, axis=0)))
            print("time weighted fitnesses: " + str(np.average(best_fitnesses, axis=0, weights=1. / best_timeses)))
            pass
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
