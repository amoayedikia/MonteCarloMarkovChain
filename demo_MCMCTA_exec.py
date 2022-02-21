
import random
import warnings
from sklearn import metrics
from sklearn.metrics import f1_score
import statistics
import numpy as np
import os
import csv
import json
import pandas as pd


def MatFact(X):
    u, s, vh = np.linalg.svd(X)
    return vh


def MatGen(t, w):
    data = []
    arr = []

    for i in range(t):
        for j in range(w):
            arr.append(0)

        data.append(arr)
        arr = []

    return np.array(data)


def DataGen():
    file = r'facts.csv'
    df = pd.read_csv(file, nrows=1200)
    questions = df['question']
    jsons = df['metadata']

    # Getting the list of raters
    ListOfRaters = []
    for items in jsons:
        outp = json.loads(items)
        judg = outp["judgments"]
        for j in judg:
            rater = j['rater']
            if rater not in ListOfRaters:
                ListOfRaters.append(rater)

    workers = len(ListOfRaters)

    # Getting the list of questions
    ListOfQues = []
    for items in jsons:
        outp = json.loads(items)
        ques = outp["question"]
        if ques not in ListOfQues:
            ListOfQues.append(ques)

    # Generating worker-task matrix
    questions = 1000  # len(ListOfQues)
    wt = MatGen(questions, workers)  # np.zeros(shape=(workers, questions))
    for q in range(0, questions):
        outp = json.loads(jsons[q])
        judgments = outp["judgments"]
        for j in judgments:
            rater = j['rater']
            ans = j['judgment']
            idx = ListOfRaters.index(rater)
            if(ans == 0):
                wt[q, idx] = 0
            else:
                wt[q, idx] = 1

    original_dt = wt
    wt = wt + np.random.rand(wt.shape[0], wt.shape[1])
    return wt, original_dt


def gaussian(X, mu, cov):
    n = X.shape[0]
    diff = (X - mu).T
    mtx = np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) *
                      np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)
    return mtx


def initialize_clusters(X, y, n_clusters):
    clusters = []
    idx = np.arange(X.shape[0])

    mu_k = y  # kmeans.cluster_centers_

    for i in range(n_clusters):
        clusters.append({
            'pi_k': 1.0 / n_clusters,
            'mu_k': mu_k[i],
            'cov_k': np.identity(X.shape[1], dtype=np.float64)
        })

    return clusters


def expectation_step(X, clusters):
    totals = np.zeros((X.shape[0], 1), dtype=np.float64)

    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']
        gamma_nk = (pi_k * gaussian(X, mu_k, cov_k))  # .astype(np.float64)

        for i in range(X.shape[0]):
            totals[i] += gamma_nk[i]

        cluster['gamma_nk'] = gamma_nk
        cluster['totals'] = totals

    for cluster in clusters:
        cluster['gamma_nk'] /= cluster['totals']

    return clusters


def maximization_step(X, clusters):
    N = float(X.shape[0])

    for cluster in clusters:
        gamma_nk = cluster['gamma_nk']
        cov_k = np.zeros((X.shape[1], X.shape[1]))

        N_k = np.sum(gamma_nk, axis=0)

        pi_k = N_k / N
        mu_k = np.sum(gamma_nk * X, axis=0) / N_k

        for j in range(X.shape[0]):
            diff = (X[j] - mu_k).reshape(-1, 1)
            cov_k += gamma_nk[j] * np.dot(diff, diff.T)

        cov_k /= N_k

        cluster['pi_k'] = pi_k
        cluster['mu_k'] = mu_k
        cluster['cov_k'] = cov_k

    return cluster['mu_k']


def prior(x):
    #x[0] = mu, x[1]=sigma (new or current)
    # returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    # returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    # It makes the new sigma infinitely unlikely.
    if(x[1] <= 0):
        return 0
    return 1

# Computes the likelihood of the data given a sigma (new or current) according to equation (2)


def manual_log_like_normal(x, data):
    #x[0]=mu, x[1]=sigma (new or current)
    # data = the observation
    return np.sum(-np.log(x[1] * np.sqrt(2 * np.pi))-((data-x[0])**2) / (2*x[1]**2))

# Same as manual_log_like_normal(x,data), but using scipy implementation. It's pretty slow.


def log_lik_normal(x, data):
    #x[0]=mu, x[1]=sigma (new or current)
    # data = the observation
    return np.sum(np.log(scipy.stats.norm(x[0], x[1]).pdf(data)))


# Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return (accept < (np.exp(x_new-x)))


def metropolis_hastings(likelihood_computer, prior, transition_model, param_init, iterations, data, acceptance_rule):
    # likelihood_computer(x,data): returns the likelihood that these parameters generated the data
    # transition_model(x): a function that draws a sample from a symmetric distribution and returns it
    # param_init: a starting sample
    # iterations: number of accepted to generated
    # data: the data that we wish to model
    # acceptance_rule(x,x_new): decides whether to accept or reject the new sample

    x = param_init
    accepted = []
    rejected = []
    for i in range(iterations):
        x_new = transition_model(x)
        x_lik = likelihood_computer(x, data)
        x_new_lik = likelihood_computer(x_new, data)
        if (acceptance_rule(x_lik + np.log(prior(x)), x_new_lik+np.log(prior(x_new)))):
            x = x_new
            accepted.append(x_new[1])
        else:
            rejected.append(x_new[1])

    return np.array(accepted), np.array(rejected)


def DataGenerator(w, t):
    data = []  # np.matrix([[0 for xx in range(t)] for yy in range(w)])
    arr = []

    for i in range(w):
        for j in range(t):
            num = np.random.uniform(0, 1)
            if (num > 0.5):
                arr.append(1)
            else:
                arr.append(0)

        data.append(arr)
        arr = []

    return np.array(data)


warnings.filterwarnings("ignore")
Results = pd.DataFrame(columns=["FScore", "AUC"])

wt, original_dt = DataGen()

p = 0
acc_count = 0
rej_count = 0
acc_pop = []
rej_pop = []
population = []

status = "INIT"
X = wt[:, :]
all_tasks = X.shape[0]
all_workers = X.shape[1]
t = (65 * 1)
X = X[:t, :]
all_tasks = 650
population = []
_BreakPoint = 0.8
SpamRatio = 0.9

if(status == "JOB"):
    while(1):
        if(len(population) < (57 * SpamRatio)):
            rnd = np.random.random()
            if(0 < rnd and rnd <= _BreakPoint):
                population.append(rnd)
        else:
            rnd = np.random.random()
            if(rnd > _BreakPoint):
                population.append(rnd)
                if(len(population) == 57):
                    random.shuffle(population)
                    print(population)
                    print(np.mean(population))
                    break

for task in range((t + 1), all_tasks):

    if(status == "INIT"):
        mu = np.mean(X, axis=0)

        cov = np.dot((X - mu).T, X - mu) / (X.shape[0] - 1)
        y = gaussian(X, mu=mu, cov=cov)
        n_clust = len(X)

        clusters = initialize_clusters(X, y, n_clust)
        clusters = expectation_step(X, clusters)
        mu_s = maximization_step(X, clusters)
        psize = len(mu_s)
        population = mu_s
        print(population)
        print(np.mean(population))

        # MCMC #The tranistion model defines how to move from sigma_current to sigma_new
        def transition_model(x): return [
            x[0], np.random.normal(x[1], 0.5, (1,))]

        for p in range(0, psize):
            observation = population[p]
            mu_obs = observation.mean()
            accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model, [
                                                     mu_obs, 0.1], 1, observation, acceptance)
            if(len(accepted) > 0):
                acc_count = acc_count + accepted
                acc_pop.append(p)

            if(len(rejected) > 0):
                rej_count = rej_count + rejected
                rej_pop.append(p)

        aa = MatGen(1, all_workers)
        for a in acc_pop:
            aa[0, a] = 1
        for r in rej_pop:
            aa[0, r] = 0

        X = np.vstack((X, aa))

        status = "JOB"

    else:

        # print("\n Processing task ", task)
        acc_pop = []
        rej_pop = []
        mu = np.mean(X, axis=0)
        cov = np.dot((X - mu).T, X - mu) / (X.shape[0] - 1)
        y = gaussian(X, mu=mu, cov=cov)
        n_clust = len(X)

        clusters = initialize_clusters(X, y, n_clust)
        clusters = expectation_step(X, clusters)
        population = maximization_step(X, clusters)
        psize = len(population)
        # population = mu_s

        # MCMC #The tranistion model defines how to move from sigma_current to sigma_new
        def transition_model(x): return [
            x[0], np.random.normal(x[1], 0.5, (1,))]

        for p in range(0, psize):
            observation = population[p]
            mu_obs = observation.mean()
            accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model, [
                                                     mu_obs, 0.1], 1, observation, acceptance)
            if(len(accepted) > 0):
                acc_count = acc_count + accepted
                acc_pop.append(p)

            if(len(rejected) > 0):
                rej_count = rej_count + rejected
                rej_pop.append(p)

        aa = MatGen(1, all_workers)  # np.zeros(shape=(1, all_workers))

        for a in acc_pop:
            aa[0, a] = 1
        for r in rej_pop:
            aa[0, r] = 0
        X = np.vstack((X, aa))

        if(len(acc_pop) > 0):
            f1 = metrics.f1_score(
                original_dt[task, :], np.transpose(aa), average="macro")
            fp, tp, thresholds = metrics.roc_curve(
                original_dt[task, :], np.transpose(aa), pos_label=1)
            auc = metrics.auc(fp, tp)
            df = pd.DataFrame({"FScore": [f1], "AUC": [auc]})
            Results = Results.append(df)
            # print(Results)
            # print("F-score and AUC of task: ", task, " are: ",  f1, " and ", auc)

avg_fscore = np.nanmean(Results["FScore"])
avg_auc = np.nanmean(Results["AUC"])

std_fscore = np.std(Results["FScore"])
std_auc = np.std(Results["AUC"])

max_fscore = np.max(Results["FScore"])
max_auc = np.max(Results["AUC"])


print("\n std of fscore , auc \n")
print(std_fscore)
print(std_auc)

print("\n")

print("max fscore , auc \n")
print(max_fscore)
print(max_auc)

print("\n")

print("average fscore , auc \n")
print(avg_fscore)
print(avg_auc)

########
