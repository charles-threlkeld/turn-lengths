import pandas
import pymc3

turn_lengths_fname = "tcu_lengths.csv"
data = pandas.read_csv(turn_lengths_fname)
data["one_less_word"] = data["tcu_words"] - 1
data["two_less_words"] = data["tcu_words"] - 2
data["one_less_syllable"] = data["tcu_syllables"] - 1
data["two_less_syllables"] = data["tcu_syllables"] - 2
data["three_less_syllables"] = data["tcu_syllables"] - 3


observed_data = np.asarray(data[data["two_less_words"] > 0]["two_less_words"])
#observed_data = np.asarray(data["tcu_words"])


# We're fitting continuous models
# For other options for distributions,
# see https://docs.pymc.io/en/v3/api/distributions/continuous.html
with pymc3.Model() as exponential_model:
    rate = pymc3.Gamma("rate",
                       mu=0.13,
                       sd=0.4)
    turn_lengths = pymc3.Exponential("turn_lengths",
                                     lam=rate,
                                     observed=observed_data)
    exponential_trace = pymc3.sample(10000,
                                     tune=4000,
                                     target_accept=0.9,
                                     return_inferencedata=False)
    exponential_posterior = \
        pymc3.sampling.sample_posterior_predictive(exponential_trace,
                                                   keep_size=True)

with pymc3.Model() as gamma_model:
    rate = pymc3.Gamma("rate",
                       mu=0.13,
                       sd=0.4)
    shape = pymc3.Gamma("shape",
                        mu=1,
                        sd=0.4)
    turn_lengths = pymc3.Gamma("turn_lengths",
                               alpha=shape,
                               beta=rate,
                               observed=observed_data)
    gamma_trace = pymc3.sample(10000,
                                 tune=4000,
                                 target_accept=0.9,
                                 return_inferencedata=False)
    gamma_posterior = \
        pymc3.sampling.sample_posterior_predictive(gamma_trace,
                                                   keep_size=True)


# Arviz is the main package for analyzing pymc objects after fitting
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt

az.style.use("arviz-darkgrid")

# We'll cut this off for a better view
graph_data = data[(data["two_less_words"] > 0) & (data["two_less_words"] < 30)]["two_less_words"]

# Plot_dist will choose between a kernel-density estimate or discrete version
# Depending on the data
az.plot_dist(graph_data, label="Data", kind="hist",
             hist_kwargs={"bins": 20, "align": "right"},
             color=sns.color_palette()[1])
az.plot_dist(gamma_posterior["turn_lengths"],
             label="Gamma Model",
             color=sns.color_palette()[0])
az.plot_dist(exponential_posterior["turn_lengths"],
             label="Exponential Model",
             color=sns.color_palette()[2])
plt.xlim(1, 30)
plt.title("Turn Construction Unit Word Count")
plt.xlabel("Number of Words")
plt.xticks(ticks=list(range(1, 30, 5)),
           labels=list(range(1, 30, 5)))
plt.tick_params(axis="y", labelleft=False, left=False)
plt.ylabel("Density")
plt.show()

# # Print details of the gamma model.
# # The shape is particularly interesting since 1 falls withing the high-density interval,
# # meaning that we can't rule our an exponential curve fitting just as well
# # Details at: https://python.arviz.org/en/latest/api/generated/arviz.summary.html
# print(az.summary(gamma_trace))

# # We'll plot the gamma shape posterior, just to see what we're working with
# az.plot_dist(gamma_trace["shape"])
# plt.show()

# # Compare the two models.
# # The WAIC is an error metric
# # Using the deviance score means that lower is better
# # WAIC accounts for "effective variables", so we don't expect the exponential to do much
# # better than the gamma
# # Details at: https://python.arviz.org/en/latest/api/generated/arviz.compare.html
print(az.compare({"Gamma": gamma_trace,
                  "Exponential": exponential_trace},
                 ic="waic", scale="deviance"))


####################################
# EXPERIMENTAL BAYES FACTORS BELOW #
####################################

from pymc3.model import modelcontext
import scipy.stats as st
from numpy import dot
from scipy.linalg import cholesky as chol
import warnings
import numpy as np

def marginal_lik(mtrace, model=None, logp=None, maxiter=10000):
    """The Bridge Sampling Estimator of the Marginal Likelihood.

    Parameters
    ----------
    mtrace : MultiTrace, result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    logp : Model log-probability function, read from the model by default
    maxiter : Maximum number of iterations

    Returns
    -------
    marg_lik : Estimated Marginal log-likelihood
    """

    r0, tol1, tol2 = 0.5, 1e-10, 1e-4

    model = modelcontext(model)
    if logp is None:
        logp = model.logp_array
    vars = model.free_RVs

    # Split the samples into two parts
    # Use the first 50% for fitting the proposal distribution and the second 50%
    # in the iterative scheme.
    len_trace = len(mtrace)
    nchain = mtrace.nchains

    N1_ = len_trace // 2
    N1 = N1_*nchain
    N2 = len_trace*nchain - N1

    neff_list = dict() # effective sample size, a dict of ess for each var

    arraysz = model.bijection.ordering.size
    samples_4_fit = np.zeros((arraysz, N1))
    samples_4_iter = np.zeros((arraysz, N2))

    # matrix with already transformed samples.
    for var in vars:
        varmap = model.bijection.ordering.by_name[var.name]
        # for fitting the proposal
        x = mtrace[:N1_][var.name] # Getting N1 samples from each chain
        samples_4_fit[varmap.slc, : ] = x.reshape((x.shape[0],
                                                   np.prod(x.shape[1:], dtype=int))).T # What is .slc? I guess gives indices for the variables??
        # for the iterative scheme
        x2 = mtrace[N1_:][var.name]
        samples_4_iter[varmap.slc, :] = x2.reshape((x2.shape[0],
                                                    np.prod(x2.shape[1:], dtype=int))).T
        # effective sample size of samples_4_iter, scalar
        neff_list.update({var.name: az.ess(x)})

    # median effective sample size (scalar)
    neff = np.median(list(neff_list.values()))

    # get mean & covariance matrix and generate samples from proposal
    m = np.mean(samples_4_fit, axis=1)
    V = np.cov(samples_4_fit)
    L = chol(V, lower=True)

    # Draw N2 samples from the proposal distribution
    gen_samples = m[:, None] + dot(L, st.norm.rvs(0, 1,
                                                  size=samples_4_iter.shape))

    # Evaluate proposal distribution for posterior & generated samples
    q12 = st.multivariate_normal.logpdf(samples_4_iter.T, m, V)
    q22 = st.multivariate_normal.logpdf(gen_samples.T, m, V)

    # Evaluate unnormalized posterior for posterior & generated sampels
    q11 = np.asarray([logp(point) for point in samples_4_iter.T])
    q21 = np.asarray([logp(point) for point in gen_samples.T])

    # Iterative scheme as proposed in Meng and Wong (1996) to estimate
    # the marginal likelihood
    def iterative_scheme(q11, q12, q21, q22, r0, neff, tol, maxiter, criterion):
        l1 = q11 - q12
        l2 = q21 - q22
        lstar = np.median(l1) # To increase numerical stability
                              # subtracting the median of l1 from l1 & l2 later
        s1 = neff/(neff + N2)
        s2 = N2/(neff + N2)

        r = r0
        r_vals = [r]
        logml = np.log(r) + lstar
        criterion_val = 1 + tol

        i = 0
        while (i <= maxiter) & (criterion_val > tol):
            rold = r
            logmlold = logml
            numi = np.exp(l2 - lstar)/(s1 * np.exp(l2 - lstar) + s2 * r)
            deni = 1/(s1 * np.exp(l1 - lstar) + s2 * r)
            if np.sum(~np.isfinite(numi))+np.sum(~np.isfinite(deni)) > 0:
                warnings.warn("""Infinite value in iterative scheme, returning NaN.
                Try rerunning with more samples.""")
            r = (N1/N2) * np.sum(numi)/np.sum(deni)
            r_vals.append(r)
            logml = np.log(r) + lstar
            i += 1
            if criterion == 'r':
                criterion_val = np.abs((r - rold)/r)
            elif criterion== 'logml':
                criterion_val = np.abs((logml - logmlold)/logml)

        if i >= maxiter:
            return dict(logml = np.Nan, niter = i, r_vals = np.asarray(r_vals))
        else:
            return dict(logml = logml, niter = i)

    # Run iterative scheme:
    tmp = iterative_scheme(q11, q12, q21, q22, r0, neff, tol1, maxiter, 'r')
    if ~np.isfinite(tmp['logml']):
        warnings.warn("""logml could not be estimated within maxiter, rerunning with
                      adjusted starting value. Estimate might be more variable than usual.""")
        # use geometric mean as starting value
        r0_2 = np.sqrt(tmp['r_vals'][-2]*tmp['r_vals'][-1])
        tmp = iterative_scheme(q11, q12, q21, q22, r0_2, neff, tol2, maxiter, 'logml')

    return dict(logml = tmp['logml'], niter = tmp['niter'], method = "normal",
                q11 = q11, q12 = q12, q21 = q21, q22 = q22)


def bayes_factor(trace1, trace2, model1, model2):
    """
    A bridge sampling estimate of the Bayes factor p(y|model1) / p(y|model2)

    Parameters:
    -----------
    trace1 : an InferenceData object
        Posterior samples from model1
    trace2 : an InferenceData object
        Posterior samples from model2
    model1 : PyMC model
    model2 : PyMC model

    Returns:
    --------
    The estimated Bayes Factor (scalar)

    Notes:
    ------
    The user needs to ensure the traces are sampled using the same observed data
    """

    log_marginal_likelihood1 = marginal_lik(model=model1, mtrace=trace1)['logml']
    log_marginal_likelihood2 = marginal_lik(model=model2, mtrace=trace2)['logml']
    return np.exp(log_marginal_likelihood1 - log_marginal_likelihood2)

# Calling the Bayes Factor code for our models:
BF = bayes_factor(exponential_trace,
                  gamma_trace,
                  exponential_model,
                  gamma_model)
BF2 = bayes_factor(gamma_trace,
                  exponential_trace,
                  gamma_model,
                  exponential_model)
print(f"The data is {BF} times more likely under the exponential model than the gamma model")
print(f"or {BF2} times more likely under the gamma model than the exponential model")
