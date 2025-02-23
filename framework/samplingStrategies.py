import random
import itertools
import numpy as np
import networkx as ntx

from tqdm import tqdm
from scipy import stats, optimize
from .stratificationStrategies import stratifyCSRF


def clusterCostFunction(heads, triples, c1=45, c2=25):
    """
    Compute the cluster-based annotation cost function (in hours)

    :param heads: num of heads (clusters)
    :param triples: num of triples
    :param c1: average cost for Entity Identification (EI)
    :param c2: average cost for Fact Verification (FV)
    :return: the annotation cost function (in hours)
    """

    return (heads * c1 + triples * c2) / 3600


class SRSSampler(object):
    """
    This class represents the Simple Random Sampling (SRS) scheme used to perform KG accuracy evaluation.
    The SRS estimator is an unbiased estimator.
    """

    def __init__(self, alpha=0.05, a_prior=1, b_prior=1):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability αlpha/2

        :param alpha: the user defined confidence level
        :param a_prior: the user defined alpha prior for bayesian credible intervals
        :param b_prior: the user defined beta prior for bayesian credible intervals
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha/2)

        # bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

        # possible confidence intervals
        self.computeMoE = {'wald': self.computeSCI, 'wilson': self.computeWCI, 'bayesET': self.computeBET, 'bayesHPD': self.computeBHPD}

    def updatePriors(self, a_prior, b_prior):
        """
        Update alpha and beta priors for bayesian credible intervals

        :param a_prior: the new alpha prior
        :param b_prior: the new beta prior
        """

        # update bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

    def estimate(self, sample):
        """
        Estimate the KG accuracy based on sample

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: KG accuracy estimate
        """

        return sum(sample)/len(sample)

    def computeVar(self, sample):
        """
        Compute the sample variance

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: sample variance
        """

        # estimate mean
        ae = self.estimate(sample)
        # count number of clusters in sample
        n = len(sample)
        # compute variance
        var = (1/n) * (ae * (1-ae))
        return var

    def computeSCI(self, sample):
        """
        Compute the Standard Confidence Interval (CI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)
        # compute sample variance
        var = self.computeVar(sample)

        # compute the margin of error (i.e., z * sqrt(var))
        moe = self.z * (var ** 0.5)
        # compute upper and lower bounds given estimate and MoE
        lowerB = ae - moe
        upperB = ae + moe

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeWCI(self, sample):
        """
        Compute the Wilson Confidence Interval (CI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)

        n = len(sample)
        x = sum(sample)

        # compute the adjusted sample size
        n_ = n + self.z ** 2
        # compute the adjusted number of successes
        x_ = x + (self.z ** 2) / 2
        # compute the adjusted mean estimate
        ae_ = x_ / n_

        # compute the margin of error
        moe = ((self.z * (n ** 0.5)) / n_) * (((ae * (1 - ae)) + ((self.z ** 2) / (4 * n))) ** 0.5)

        if (n <= 50 and x in [1, 2]) or (n >= 51 and x in [1, 2, 3]):
            lowerB = 0.5 * stats.chi2.isf(q=1 - self.alpha, df=2 * x) / n
        else:
            lowerB = max(0, ae_ - moe)

        if (n <= 50 and x in [n - 1, n - 2]) or (n >= 51 and x in [n - 1, n - 2, n - 3]):
            upperB = 1 - (0.5 * stats.chi2.isf(q=1 - self.alpha, df=2 * (n - x))) / n
        else:
            upperB = min(1, ae_ + moe)

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeBET(self, sample):
        """
        Compute the Bayesian Equal-Tailed Credible Interval (CrI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the CrI as (lowerBound, upperBound)
        """

        # compute sample size and number of successes
        n = len(sample)
        x = sum(sample)

        # compute posterior params -- posterior distr is Beta since prior is Beta and likelihood is Binom
        a_post = self.a_prior + x
        b_post = self.b_prior + n - x

        # define Beta posterior distr
        posterior = stats.beta(a_post, b_post)

        if x == 0:
            lowerB = 0
            upperB = 1 - (self.alpha/2)**(1/n)
        elif x == 1:
            lowerB = 0
            upperB = posterior.ppf(1-self.alpha/2)
        elif x == n-1:
            lowerB = posterior.ppf(self.alpha/2)
            upperB = 1
        elif x == n:
            lowerB = (self.alpha/2)**(1/n)
            upperB = 1
        else:
            lowerB = posterior.ppf(self.alpha/2)
            upperB = posterior.ppf(1-self.alpha/2)

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeBHPD(self, sample):
        """
        Compute the Bayesian Highest Posterior Density Credible Interval (CrI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the CrI as (lowerBound, upperBound)
        """

        # compute sample size and number of successes
        n = len(sample)
        x = sum(sample)

        # compute posterior params -- posterior distr is Beta since prior is Beta and likelihood is Binom
        a_post = self.a_prior + x
        b_post = self.b_prior + n - x

        # define Beta posterior distr
        posterior = stats.beta(a_post, b_post)

        if x == 0:  # posterior distr has mode in x = 0 -- HPD interval is (0, q(1-alpha))
            return 0, posterior.ppf(1 - self.alpha)

        if x == n:  # posterior distr has mode in x = n -- HPD interval is (q(alpha), 1)
            return posterior.ppf(self.alpha), 1

        # compute credible mass
        mass = 1 - self.alpha

        # objective function to minimize -- i.e. interval width
        def objective(params):
            lower, upper = params
            return upper - lower

        # constraint: interval should contain credible mass
        def constraint(params):
            lower, upper = params
            return posterior.cdf(upper) - posterior.cdf(lower) - mass

        # initial guess for the interval -- rely on corresponding ET CrI (w/o ad hoc changes)
        guess = posterior.ppf([self.alpha/2, 1-self.alpha/2])

        # minimize the width of the interval (objective) subject to the credible mass (constraint)
        res = optimize.minimize(objective, guess, constraints={'type': 'eq', 'fun': constraint}, bounds=[(0, 1), (0, 1)])

        if res.success:
            return res.x
        else:  # optimization failed
            raise RuntimeError('optimization failed')

    def run(self, kg, groundTruth, minSample=30, thrMoE=0.05, ciMethod='wald', c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ SRS and stop when MoE < thr
        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param ciMethod: the method used to compute Confidence Intervals (CIs)
        :param c1: average cost for Entity Identification (EI)
        :param c2: average cost for Fact Verification (FV)
        :param iters: number of times the estimation is performed
        :return: evaluation statistics
        """

        if self.a_prior == -1 and self.b_prior == -1 and ciMethod in ['bayesET', 'bayesHPD']:
            adaptive = True
        else:
            adaptive = False

        # collect the sample (upper bound) for every iteration
        sample4iter = [random.choices(population=kg, k=1000) for _ in range(iters)]  # k=1000 enough

        estimates = []
        for j in tqdm(range(iters)):  # perform iterations
            # set params
            lowerB = 0.0
            upperB = 1.0
            heads = {}
            sample = []

            while (upperB-lowerB)/2 > thrMoE:  # stop when MoE gets lower than threshold
                # perform SRS over the KG
                id_, triple = sample4iter[j].pop(0)  # random.choices(population=kg, k=1)[0]
                if triple[0] not in heads:  # found new head (cluster) -- increase the num of clusters within sample
                    heads[triple[0]] = 1
                # get annotations for triples within sample
                sample += [groundTruth[id_]]

                if len(sample) >= minSample:  # compute CI
                    if adaptive:  # conduct adaptive strategy
                        # set temporary interval boundaries
                        lowerMin = 0.0
                        upperMin = 1.0
                        for prior in [1/3, 1/2, 1]:  # iterate over non-informative priors
                            # update prior
                            self.updatePriors(prior, prior)
                            # compute interval given prior
                            lowerCurrent, upperCurrent = self.computeMoE[ciMethod](sample)
                            if upperCurrent-lowerCurrent < upperMin-lowerMin:  # computed interval has smaller width than min -- update
                                lowerMin = lowerCurrent
                                upperMin = upperCurrent
                        # set interval w/ minimal boundaries
                        lowerB = lowerMin
                        upperB = upperMin
                    else:
                        lowerB, upperB = self.computeMoE[ciMethod](sample)

            # compute cost function (cluster based)
            cost = clusterCostFunction(len(heads), len(sample), c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(sample)

            # store stats
            estimates += [[len(sample), estimate, cost, lowerB, upperB]]
        # return stats
        return estimates


class TWCSSampler(object):
    """
    This class represents the Two-stage Weighted Cluster Sampling (TWCS) scheme used to perform KG accuracy evaluation.
    The TWCS estimator is an unbiased estimator.
    """

    def __init__(self, alpha=0.05, a_prior=1, b_prior=1):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability α/2

        :param alpha: the user defined confidence level
        :param a_prior: the user defined alpha prior for bayesian credible intervals
        :param b_prior: the user defined beta prior for bayesian credible intervals
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha/2)

        # bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

        # possible confidence intervals
        self.computeMoE = {'wald': self.computeSCI, 'wilson': self.computeWCI, 'bayesET': self.computeBET, 'bayesHPD': self.computeBHPD}

    def updatePriors(self, a_prior, b_prior):
        """
        Update alpha and beta priors for bayesian credible intervals

        :param a_prior: the new alpha prior
        :param b_prior: the new beta prior
        """

        # update bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

    def estimate(self, sample):
        """
        Estimate the KG accuracy based on the sample

        :param sample: input sample (i.e., clusters of triples) used for estimation
        :return: KG accuracy estimate
        """

        # compute, for each cluster, the cluster accuracy estimate
        cae = [sum(cluster)/len(cluster) for cluster in sample]
        # compute estimate
        return sum(cae)/len(cae)

    def computeVar(self, sample):
        """
        Compute the sample variance

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: sample variance
        """

        # compute, for each cluster, the cluster accuracy estimate
        cae = [sum(cluster) / len(cluster) for cluster in sample]
        # compute estimate
        ae = sum(cae) / len(cae)

        # count number of clusters in sample
        n = len(sample)

        if n*(n-1) != 0:  # compute variance
            var = (1/(n*(n-1)))*sum([(cae[i] - ae) ** 2 for i in range(n)])
        else:  # set variance to inf
            var = np.inf
        return var

    def computeESS(self, sample, numT):
        """
        Compute the Effective Sample Size adjusted for the design degrees of freedom

        :param sample: input sample (i.e., set of triples) used for estimation
        :param numT: total number of triples in the sample
        :return: the effective sample size
        """

        # compute clusters mean size
        meanSize = np.mean([len(cluster) for cluster in sample])

        N = len(sample)
        M = sum([len(c) for c in sample])

        x_ = 1 / M * sum([sum(c) for c in sample])

        csizes = [len(c) for c in sample]
        maxsize = max(csizes)

        part1 = sum([sum([(c[i] - x_) ** 2 for c in sample if i < len(c)]) for i in range(maxsize)])
        s = 1 / (M - 1) * part1

        if s == 0:
            return numT

        icc = meanSize / (meanSize - 1) * (1 / (s * N)) * (sum([(sum(c) / len(c) - x_) ** 2 for c in sample])) - 1 / (meanSize - 1)

        dEffect = 1 + ((meanSize - 1) * icc)

        if dEffect < 1e-06:  # tolerance threshold to avoid numerical instability -- dEffect set to zero when < 1e-06
            dEffect = 0

        # compute the Effective Sample Size (ESS)
        if dEffect > 0:
            ess = (numT / dEffect)
        else:
            ess = numT

        # return effective sample size
        return ess

    def computeSCI(self, sample, numT):
        """
        Compute the Standard Confidence Interval (CI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)
        # compute sample variance
        var = self.computeVar(sample)

        # compute the margin of error (i.e., z * sqrt(var))
        moe = self.z * (var ** 0.5)
        # compute upper and lower bounds given estimate and MoE
        lowerB = ae - moe
        upperB = ae + moe

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeWCI(self, sample, numT):
        """
        Compute the Wilson Confidence Interval (CI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :param numT: total number of triples in the sample
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)

        # compute effective sample size
        n = self.computeESS(sample, numT)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n*ae
        # compute the adjusted sample size
        n_ = n + self.z ** 2
        # compute the adjusted number of successes
        x_ = x + (self.z ** 2)/2
        # compute the adjusted mean estimate
        ae_ = x_/n_

        # compute the margin of error
        moe = ((self.z * (n ** 0.5)) / n_) * (((ae * (1-ae)) + ((self.z ** 2) / (4 * n))) ** 0.5)

        if (n <= 50 and x in [1, 2]) or (n >= 51 and x in [1, 2, 3]):
            lowerB = 0.5 * stats.chi2.isf(q=1-self.alpha, df=2 * x) / n
        else:
            lowerB = max(0, ae_ - moe)

        if (n <= 50 and x in [n - 1, n - 2]) or (n >= 51 and x in [n - 1, n - 2, n - 3]):
            upperB = 1 - (0.5 * stats.chi2.isf(q=1-self.alpha, df=2 * (n-x))) / n
        else:
            upperB = min(1, ae_ + moe)

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeBET(self, sample, numT):
        """
        Compute the Bayesian Equal-Tailed Credible Interval (CrI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :param numT: total number of triples in the sample
        :return: the CrI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)

        # compute effective sample size
        n = self.computeESS(sample, numT)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n * ae

        # compute posterior params -- posterior distr is Beta since prior is Beta and likelihood is Binom
        a_post = self.a_prior + x
        b_post = self.b_prior + n - x

        # define Beta posterior distr
        posterior = stats.beta(a_post, b_post)

        if x == 0:
            lowerB = 0
            upperB = 1 - (self.alpha/2)**(1/n)
        elif x == 1:
            lowerB = 0
            upperB = posterior.ppf(1-self.alpha/2)
        elif x == n-1:
            lowerB = posterior.ppf(self.alpha/2)
            upperB = 1
        elif x == n:
            lowerB = (self.alpha/2)**(1/n)
            upperB = 1
        else:
            lowerB = posterior.ppf(self.alpha/2)
            upperB = posterior.ppf(1-self.alpha/2)

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeBHPD(self, sample, numT):
        """
        Compute the Bayesian Highest Posterior Density Credible Interval (CrI)

        :param sample: input sample (i.e., set of triples) used for estimation
        :param numT: total number of triples in the sample
        :return: the CrI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(sample)

        # compute effective sample size
        n = self.computeESS(sample, numT)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n * ae

        # compute posterior params -- posterior distr is Beta since prior is Beta and likelihood is Binom
        a_post = self.a_prior + x
        b_post = self.b_prior + n - x

        # define Beta posterior distr
        posterior = stats.beta(a_post, b_post)

        if x == 0:  # posterior distr has mode in x = 0 -- HPD interval is (0, q(1-alpha))
            return 0, posterior.ppf(1 - self.alpha)

        if x == n:  # posterior distr has mode in x = n -- HPD interval is (q(alpha), 1)
            return posterior.ppf(self.alpha), 1

        # compute credible mass
        mass = 1 - self.alpha

        # objective function to minimize -- i.e. interval width
        def objective(params):
            lower, upper = params
            return upper - lower

        # constraint: interval should contain credible mass
        def constraint(params):
            lower, upper = params
            return posterior.cdf(upper) - posterior.cdf(lower) - mass

        # initial guess for the interval -- rely on corresponding ET CrI (w/o ad hoc changes)
        guess = posterior.ppf([self.alpha/2, 1-self.alpha/2])

        # minimize the width of the interval (objective) subject to the credible mass (constraint)
        res = optimize.minimize(objective, guess, constraints={'type': 'eq', 'fun': constraint}, bounds=[(0, 1), (0, 1)])

        if res.success:
            return res.x
        else:  # optimization failed
            raise RuntimeError('optimization failed')

    def run(self, kg, groundTruth, stageTwoSize=5, minSample=30, thrMoE=0.05, ciMethod='wald', c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ TWCS and stop when MoE < thr
        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param stageTwoSize: second-stage sample size.
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param ciMethod: the method used to compute Confidence Intervals (CIs)
        :param c1: average cost for Entity Identification (EI)
        :param c2: average cost for Fact Verification (FV)
        :param iters: number of times the estimation is performed
        :return: evaluation statistics
        """

        if self.a_prior == -1 and self.b_prior == -1 and ciMethod in ['bayesET', 'bayesHPD']:
            adaptive = True
        else:
            adaptive = False

        # prepare (head) clusters
        clusters = {}
        for id_, triple in kg:  # iterate over KG triples and make clusters
            if triple[0] in clusters:  # cluster found -- add triple (id)
                clusters[triple[0]] += [id_]
            else:  # cluster not found -- create cluster and add triple (id)
                clusters[triple[0]] = [id_]

        # get cluster heads
        heads = list(clusters.keys())
        # get cluster sizes
        sizes = [len(clusters[s]) for s in heads]
        # compute cluster weights based on cluster sizes
        weights = [sizes[i]/sum(sizes) for i in range(len(sizes))]

        # collect the sample (upper bound) for every iteration
        heads4iter = [random.choices(population=heads, weights=weights, k=1000) for _ in range(iters)]  # k=1000 enough
        sample4iter = [[random.sample(clusters[h], min(stageTwoSize, len(clusters[h]))) for h in heads] for heads in heads4iter]

        estimates = []
        for j in tqdm(range(iters)):  # perform iterations
            # set params
            lowerB = 0.0
            upperB = 1.0
            numC = 0
            numT = 0
            sample = []

            while (upperB-lowerB)/2 > thrMoE:  # stop when MoE gets lower than threshold
                # increase heads number
                numC += 1

                # second-stage sampling
                stageTwo = sample4iter[j].pop(0)  # random.sample(pool, min(stageTwoSize, len(pool)))

                # get annotations for triples within sample
                sample += [[groundTruth[triple] for triple in stageTwo]]
                # increase triples number
                numT += len(stageTwo)

                if numT >= minSample:  # compute MoE
                    if adaptive:  # conduct adaptive strategy
                        # set temporary interval boundaries
                        lowerMin = 0.0
                        upperMin = 1.0
                        for prior in [1/3, 1/2, 1]:  # iterate over non-informative priors
                            # update prior
                            self.updatePriors(prior, prior)
                            # compute interval given prior
                            lowerCurrent, upperCurrent = self.computeMoE[ciMethod](sample, numT)
                            if upperCurrent-lowerCurrent < upperMin-lowerMin:  # computed interval has smaller width than min -- update
                                lowerMin = lowerCurrent
                                upperMin = upperCurrent
                        # set interval w/ minimal boundaries
                        lowerB = lowerMin
                        upperB = upperMin
                    else:
                        lowerB, upperB = self.computeMoE[ciMethod](sample, numT)

            # compute cost function (cluster based)
            cost = clusterCostFunction(numC, numT, c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(sample)

            # store stats
            estimates += [[numT, estimate, cost, lowerB, upperB]]
        # return stats
        return estimates


class STWCSSampler(object):
    """
    This class represents the Stratified Two-stage Weighted Cluster Sampling (STWCS) scheme used to perform KG accuracy evaluation.
    The STWCS estimator is an unbiased estimator.
    """

    def __init__(self, alpha=0.05, a_prior=1, b_prior=1):
        """
        Initialize the sampler and set confidence level plus Normal critical value z with right-tail probability α/2

        :param alpha: the user defined confidence level
        :param a_prior: the user defined alpha prior for bayesian credible intervals
        :param b_prior: the user defined beta prior for bayesian credible intervals
        """

        # confidence level
        self.alpha = alpha
        self.z = stats.norm.isf(self.alpha/2)

        # bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

        # instantiate the TWCS sampling method
        self.twcs = TWCSSampler(self.alpha, self.a_prior, self.b_prior)

        # possible confidence intervals
        self.computeMoE = {'wald': self.computeSCI, 'wilson': self.computeWCI, 'bayesET': self.computeBET, 'bayesHPD': self.computeBHPD}

    def updatePriors(self, a_prior, b_prior):
        """
        Update alpha and beta priors for bayesian credible intervals

        :param a_prior: the new alpha prior
        :param b_prior: the new beta prior
        """

        # update bayesian priors
        self.a_prior = a_prior
        self.b_prior = b_prior

    def estimate(self, strataSamples, strataWeights):
        """
        Estimate the KG accuracy based on the sample

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :return: KG accuracy estimate
        """

        # compute, for each stratum sample, the TWCS based accuracy estimate
        sae = [self.twcs.estimate(stratumSample) for stratumSample in strataSamples]
        # compute estimate
        return sum([sae[i] * strataWeights[i] for i in range(len(strataSamples))])

    def computeVar(self, strataSamples, strataWeights):
        """
        Compute the sample variance

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :return: sample standard deviation
        """

        # compute, for each stratum, the TWCS estimated variance
        strataVars = [self.twcs.computeVar(stratumSample) for stratumSample in strataSamples]
        # compute variance
        return sum([(strataVars[i]) * (strataWeights[i] ** 2) for i in range(len(strataSamples))])

    def computeESS(self, strataSamples, strataWeights, strataT, numC, numS):
        """
        Compute the Effective Sample Size adjusted for the design degrees of freedom

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param strataT: per-stratum number of triples in the sample
        :param numC: total number of clusters in the sample
        :param numS: total number of strata
        :return: the effective sample size
        """

        numT = sum(strataT)
        strataVars = []
        tempsDeff = []
        temps2 = []
        temps3 = []
        iccs = []
        for ix, sample in enumerate(strataSamples):
            if not sample:
                continue
            # compute clusters mean size
            meanSize = np.mean([len(cluster) for cluster in sample])

            N = len(sample)
            M = sum([len(c) for c in sample])

            x_ = (1 / M) * sum([sum(c) for c in sample])

            csizes = [len(c) for c in sample]
            maxsize = max(csizes)

            part1 = sum([sum([(c[i] - x_) ** 2 for c in sample if i < len(c)]) for i in range(maxsize)])
            if M > 1:
                s = 1 / (M - 1) * part1
            else:
                s = 0

            if s == 0 or meanSize == 1:
                icc = 0
            else:
                icc = meanSize / (meanSize - 1) * (1 / (s * N)) * (sum([(sum(c) / len(c) - x_) ** 2 for c in sample])) - 1 / (
                        meanSize - 1)

            iccs.append(icc)
            tempsDeff.append((strataWeights[ix]**2 * (numT/strataT[ix])) * (1 + ((meanSize - 1) * icc)))
            temps2.append((strataWeights[ix]**2 * (1+(meanSize-1)*icc)) * ((x_ * (1-x_)) / M))
            temps3.append(strataWeights[ix]**2 * ((x_ * (1-x_)) / M))
            strataVars.append((x_ * (1-x_)))

        var_pairs = itertools.combinations(strataVars, 2)
        var_diffs = [abs(pair[0] - pair[1]) for pair in var_pairs]
        icc_pairs = itertools.combinations(iccs, 2)
        icc_diffs = [abs(pair[0] - pair[1]) for pair in icc_pairs]
        if len(strataVars) > 1 and all([diff < 0.01 for diff in icc_diffs]) and all([diff < 0.01 for diff in var_diffs]):
            icc = np.mean(iccs)
            msize = np.mean([len(cluster) for sample in strataSamples for cluster in sample])
            dEffect = 1 + (msize - 1) * icc
        elif len(strataVars) > 1 and all([diff < 0.01 for diff in var_diffs]):
            dEffect = sum(tempsDeff)
        elif len(strataVars) > 1 and all([diff < 0.01 for diff in icc_diffs]):
            icc = np.mean(iccs)
            msize = np.mean([len(cluster) for sample in strataSamples for cluster in sample])
            ae_srs = sum([sum(c) for sample in strataSamples for c in sample]) / numT
            down = (ae_srs * (1 - ae_srs)) / numT
            dEffect = (1 + (msize - 1) * icc) * (sum(temps3)/down)
        elif len(strataVars) == 1:
            icc = np.mean(iccs)
            msize = np.mean([len(cluster) for sample in strataSamples for cluster in sample])
            dEffect = 1 + (msize -1)*icc
        else:
            up = sum(temps2)
            ae_srs = sum([sum(c) for sample in strataSamples for c in sample]) / numT
            down = (ae_srs * (1 - ae_srs)) / numT
            if down == 0:
                dEffect = 0
            else:
                dEffect = up/down

        if dEffect < 1e-06:  # tolerance threshold to avoid numerical instability -- dEffect set to zero when < 1e-06
            dEffect = 0

        # compute design factor
        dFactor = (self.z / stats.t.isf(self.alpha / 2, df=numC - numS)) ** 2

        # compute the Effective Sample Size (ESS)
        if dEffect > 0:
            ess = (numT / dEffect) * dFactor
            if ess == 0:
                ess = numT
        else:
            ess = numT

        # return effective sample size
        return ess

    def computeSCI(self, strataSamples, strataWeights, strataT, strataC):
        """
        Compute the Standard Confidence Interval (CI)

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param strataT: per-stratum number of triples in the sample
        :param strataC: per-stratum number of clusters in the sample
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(strataSamples, strataWeights)
        # compute sample variance
        var = self.computeVar(strataSamples, strataWeights)

        # compute the margin of error (i.e., z * sqrt(var))
        moe = self.z * (var ** 0.5)
        # compute upper and lower bounds given estimate and MoE
        lowerB = ae - moe
        upperB = ae + moe

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeWCI(self, strataSamples, strataWeights, strataT, strataC):
        """
        Compute the Wilson Confidence Interval (CI)

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param strataT: per-stratum number of triples in the sample
        :param strataC: per-stratum number of clusters in the sample
        :return: the CI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(strataSamples, strataWeights)

        # get number of sample triples and clusters, as well as number of strata
        numT = sum(strataT)
        numC = sum(strataC)
        numS = len(strataSamples)

        n = self.computeESS(strataSamples, strataWeights, strataT, numC, numS)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for sample in strataSamples for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n * ae
        # compute the adjusted sample size
        n_ = n + self.z ** 2
        # compute the adjusted number of successes
        x_ = x + (self.z ** 2) / 2
        # compute the adjusted mean estimate
        ae_ = x_ / n_

        # compute the margin of error
        moe = ((self.z * (n ** 0.5)) / n_) * (((ae * (1 - ae)) + ((self.z ** 2) / (4 * n))) ** 0.5)

        if (n <= 50 and x in [1, 2]) or (n >= 51 and x in [1, 2, 3]):
            lowerB = 0.5 * stats.chi2.isf(q=1 - self.alpha, df=2 * x) / n
        else:
            lowerB = max(0, ae_ - moe)

        if (n <= 50 and x in [n - 1, n - 2]) or (n >= 51 and x in [n - 1, n - 2, n - 3]):
            upperB = 1 - (0.5 * stats.chi2.isf(q=1 - self.alpha, df=2 * (n - x))) / n
        else:
            upperB = min(1, ae_ + moe)

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeBET(self, strataSamples, strataWeights, strataT, strataC):
        """
        Compute the Bayesian Equal-Tailed Credible Interval (CrI)

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param strataT: per-stratum number of triples in the sample
        :param strataC: per-stratum number of clusters in the sample
        :return: the CrI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(strataSamples, strataWeights)

        # get number of sample triples and clusters, as well as number of strata
        numT = sum(strataT)
        numC = sum(strataC)
        numS = len(strataSamples)

        n = self.computeESS(strataSamples, strataWeights, strataT, numC, numS)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for sample in strataSamples for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n * ae

        # compute posterior params -- posterior distr is Beta since prior is Beta and likelihood is Binom
        a_post = self.a_prior + x
        b_post = self.b_prior + n - x

        # define Beta posterior distr
        posterior = stats.beta(a_post, b_post)

        if x == 0:
            lowerB = 0
            upperB = 1 - (self.alpha/2)**(1/n)
        elif x == 1:
            lowerB = 0
            upperB = posterior.ppf(1-self.alpha/2)
        elif x == n - 1:
            lowerB = posterior.ppf(self.alpha/2)
            upperB = 1
        elif x == n:
            lowerB = (self.alpha/2)**(1/n)
            upperB = 1
        else:
            lowerB = posterior.ppf(self.alpha/2)
            upperB = posterior.ppf(1-self.alpha/2)

        # return CI as (lowerBound, upperBound)
        return lowerB, upperB

    def computeBHPD(self, strataSamples, strataWeights, strataT, strataC):
        """
        Compute the Bayesian Highest Posterior Density Credible Interval (CrI)

        :param strataSamples: strata samples (i.e., clusters of triples) used for estimation
        :param strataWeights: strata weights
        :param strataT: per-stratum number of triples in the sample
        :param strataC: per-stratum number of clusters in the sample
        :return: the CrI as (lowerBound, upperBound)
        """

        # compute mean estimate
        ae = self.estimate(strataSamples, strataWeights)

        # get number of sample triples and clusters, as well as number of strata
        numT = sum(strataT)
        numC = sum(strataC)
        numS = len(strataSamples)

        n = self.computeESS(strataSamples, strataWeights, strataT, numC, numS)
        if n == numT:  # effective sample size equal to actual sample size
            x = sum([sum(c) for sample in strataSamples for c in sample])
        else:  # effective sample size smaller than actual sample size
            x = n * ae

        # compute posterior params -- posterior distr is Beta since prior is Beta and likelihood is Binom
        a_post = self.a_prior + x
        b_post = self.b_prior + n - x

        # define Beta posterior distr
        posterior = stats.beta(a_post, b_post)

        if x == 0:  # posterior distr has mode in x = 0 -- HPD interval is (0, q(1-alpha))
            return 0, posterior.ppf(1 - self.alpha)

        if x == n:  # posterior distr has mode in x = n -- HPD interval is (q(alpha), 1)
            return posterior.ppf(self.alpha), 1

        # compute credible mass
        mass = 1 - self.alpha

        # objective function to minimize -- i.e. interval width
        def objective(params):
            lower, upper = params
            return upper - lower

        # constraint: interval should contain credible mass
        def constraint(params):
            lower, upper = params
            return posterior.cdf(upper) - posterior.cdf(lower) - mass

        # initial guess for the interval -- rely on corresponding ET CrI (w/o ad hoc changes)
        guess = posterior.ppf([self.alpha/2, 1-self.alpha/2])

        # minimize the width of the interval (objective) subject to the credible mass (constraint)
        res = optimize.minimize(objective, guess, constraints={'type': 'eq', 'fun': constraint}, bounds=[(0, 1), (0, 1)])

        if res.success:
            return res.x
        else:  # optimization failed
            raise RuntimeError('optimization failed')

    def run(self, kg, groundTruth, numStrata=5, stratFeature='degree', stageTwoSize=5, minSample=30, thrMoE=0.05, ciMethod='wald', c1=45, c2=25, iters=1000):
        """
        Run the evaluation procedure on KG w/ TWCS and stop when MoE < thr
        :param kg: the target KG
        :param groundTruth: the ground truth associated w/ target KG
        :param numStrata: number of considered strata
        :param stratFeature: target stratification feature.
        :param stageTwoSize: second-stage sample size
        :param minSample: the min sample size required to trigger the evaluation procedure
        :param thrMoE: the user defined MoE threshold
        :param ciMethod: the method used to compute Confidence Intervals (CIs)
        :param c1: average cost for Entity Identification (EI)
        :param c2: average cost for Fact Verification (FV)
        :param iters: number of times the estimation is performed
        :return: evaluation statistics
        """

        if self.a_prior == -1 and self.b_prior == -1 and ciMethod in ['bayesET', 'bayesHPD']:
            adaptive = True
        else:
            adaptive = False

        # prepare (head) clusters
        clusters = {}
        for id_, triple in kg:  # iterate over KG triples and make clusters
            if triple[0] in clusters:  # cluster found -- add triple (id)
                clusters[triple[0]] += [id_]
            else:  # cluster not found -- create cluster and add triple (id)
                clusters[triple[0]] = [id_]

        # get cluster heads
        heads = list(clusters.keys())

        # we consider a graph w/o parallel edges to compute degree centrality -- avoids extra boosting clusters
        g = ntx.DiGraph()
        for id, triple in kg:
            s, p, o = triple
            g.add_node(s)
            g.add_node(o)
            g.add_edge(s, o, label=p)
        # compute degree centrality
        dCent = ntx.degree_centrality(g)

        # get cluster sizes and degrees
        sizes = [len(clusters[s]) for s in heads]
        centrs = [dCent[s] for s in heads]

        # perform stratification based on stratFeature
        assert stratFeature == 'degree'
        strata = stratifyCSRF(centrs, numStrata)

        if len(strata) < numStrata:  # update number of strata
            print(f'Desired number of strata {numStrata} != obtained number of strata {len(strata)}')
            numStrata = len(strata)

        # compute strata weights
        strataWeights = [sum([sizes[i] for i in stratum])/len(kg) for stratum in strata]

        # partition data by stratum
        headsXstratum = [[heads[i] for i in stratum] for stratum in strata]
        sizesXstratum = [[sizes[i] for i in stratum] for stratum in strata]
        weightsXstratum = [[size/sum(stratumSizes) for size in stratumSizes] for stratumSizes in sizesXstratum]

        # collect the sample (upper bound) for every iteration
        strata4iter = [list(range(numStrata)) + random.choices(population=range(numStrata), weights=strataWeights, k=1000-numStrata) for _ in range(iters)]  # list(range(numStrata)) used to gather triples from each stratum -- required to handle edge cases (i.e., empy stratum after sampling)
        heads4iter = [[random.choices(population=headsXstratum[ix], weights=weightsXstratum[ix], k=1)[0] for ix in ixs] for ixs in strata4iter]
        sample4iter = [[random.sample(clusters[h], min(stageTwoSize, len(clusters[h]))) for h in heads] for heads in heads4iter]

        estimates = []
        for j in tqdm(range(iters)):  # perform iterations
            # set params
            lowerB = 0.0
            upperB = 1.0
            strataC = [0 for _ in range(numStrata)]
            strataT = [0 for _ in range(numStrata)]
            strataSamples = [[] for _ in range(numStrata)]

            while (upperB-lowerB)/2 > thrMoE:  # stop when MoE gets lower than threshold
                ix = strata4iter[j].pop(0)
                # increase heads number
                strataC[ix] += 1

                # second-stage sampling
                stageTwo = sample4iter[j].pop(0)

                # get annotations for triples within sample
                strataSamples[ix] += [[groundTruth[triple] for triple in stageTwo]]
                # increase triples number
                strataT[ix] += len(stageTwo)

                if sum(strataT) >= minSample:  # compute CI
                    if adaptive:  # conduct adaptive strategy
                        # set temporary interval boundaries
                        lowerMin = 0.0
                        upperMin = 1.0
                        for prior in [1/3, 1/2, 1]:  # iterate over non-informative priors
                            # update prior
                            self.updatePriors(prior, prior)
                            # compute interval given prior
                            lowerCurrent, upperCurrent = self.computeMoE[ciMethod](strataSamples, strataWeights, strataT, strataC)
                            if upperCurrent-lowerCurrent < upperMin-lowerMin:  # computed interval has smaller width than min -- update
                                lowerMin = lowerCurrent
                                upperMin = upperCurrent
                        # set interval w/ minimal boundaries
                        lowerB = lowerMin
                        upperB = upperMin
                    else:
                        lowerB, upperB = self.computeMoE[ciMethod](strataSamples, strataWeights, strataT, strataC)

            # compute cost function (cluster based)
            cost = clusterCostFunction(sum(strataC), sum(strataT), c1, c2)
            # compute KG accuracy estimate
            estimate = self.estimate(strataSamples, strataWeights)

            # store stats
            estimates += [[sum(strataT), estimate, cost, lowerB, upperB]]
        # return stats
        return estimates
