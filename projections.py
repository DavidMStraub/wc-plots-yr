"""WC Projections for HL LHC & Belle-II"""

from collections import defaultdict
import numpy as np
from math import sqrt
import flavio
import flavio.plots as fpl
from flavio.classes import Measurement,  Observable
from flavio.statistics.probability import MultivariateNormalDistribution
from flavio.statistics.likelihood import FastLikelihood
from wilson import Wilson
import pickle
import logging
import os
logging.basicConfig(level=logging.INFO)


SMCOV = 'smcov.p'
PDAT = 'plotdata.p'

N = 200
THREADS = 8
STEPS = 20

XMIN = -2
XMAX = 1
YMIN = -1.5
YMAX = 1.5


def tree():
    return defaultdict(tree)


def np_measurement(name, w, observables, covariance):
    """Measurement instance of `observables` measured with `covariance`
    assuming the central values to be equal to the NP predictions given
    the `Wilson` instance `w`."""
    def predict(obs):
        d = Observable.argument_format(obs, 'dict')
        return flavio.np_prediction(d.pop('name'), w, **d)
    cv = [predict(obs) for obs in observables]
    d = MultivariateNormalDistribution(cv, covariance=covariance)
    m = Measurement(name)
    m.add_constraint(observables, d)
    return m


observables = tree()
covariances = tree()
likelihoods = tree()
plotdata = tree()
npscenarios = {}

# CMS

observables['CMS']['present'] = [
    ('<P5p>(B0->K*mumu)', 1, 2),
    ('<P5p>(B0->K*mumu)', 2, 4.3),
    ('<P5p>(B0->K*mumu)', 4.3, 6),
]


observables['CMS']['Phase II'] = observables['CMS']['present']

# Building the CMS Phase II covariance matrix

# numbers provided by Sandra Malvezzi
# columns: obs, rows: unc. sources
# the last column is ignored (q^2 > 6 GeV^2)
_cms_uncorr_sys_ = np.array([[0.0025, 0.03, 0.0325, 0.0225],
[0.0065, 0.0075, 0.007, 0.0029],
[0.0095, 0.0095, 0.0095, 0.0095],
[5.25e-05, 0.0005, 0.0007, 0.0004]])
_cms_corr_sys_ = np.array([[0.0115, 0.0065, 0.0075, 0.006],
[0.02, 0.005, 0.0595, 0.053],
[0.0006, 0.0012, 0.0016, 0.0007],
[0.0014, 0.0076, 0.0108, 0.0022],
[0, 0, 0, 0.006]])
_cms_stat = np.array([0.014, 0.014, 0.009,  0.008])

covariances['CMS']['Phase II'] = (
_cms_stat[:3]**2 * np.eye(3)  # uncorrelated stat
+ np.sum(_cms_uncorr_sys_**2, axis=0)[:3] * np.eye(3)  # uncorrelated sys
+ np.sum(_cms_corr_sys_**2, axis=0)[:3] * np.ones((3, 3))  # fully correlated sys
)


# LHCb

observables['LHCb']['present'] = [
    ('<FL>(B0->K*mumu)', 1.1, 6),
    ('<S3>(B0->K*mumu)', 1.1, 6),
    ('<S4>(B0->K*mumu)', 1.1, 6),
    ('<S5>(B0->K*mumu)', 1.1, 6),
    ('<AFB>(B0->K*mumu)', 1.1, 6),
    ('<S7>(B0->K*mumu)', 1.1, 6),
    ('<S8>(B0->K*mumu)', 1.1, 6),
    ('<S9>(B0->K*mumu)', 1.1, 6),
    ('<FL>(B0->K*mumu)', 15, 19),
    ('<S3>(B0->K*mumu)', 15, 19),
    ('<S4>(B0->K*mumu)', 15, 19),
    ('<S5>(B0->K*mumu)', 15, 19),
    ('<AFB>(B0->K*mumu)', 15, 19),
    ('<S7>(B0->K*mumu)', 15, 19),
    ('<S8>(B0->K*mumu)', 15, 19),
    ('<S9>(B0->K*mumu)', 15, 19),
]

observables['LHCb']['Phase II'] = observables['LHCb']['present']

# Building the LHCb covariance matrix

# fct provided by Christoph Langenbruch
def lhcb_scale_uncertainty(lumi):
    """returns scale from luminosity for statistical uncertainty"""
    if (lumi < 3.0): #still run 1 and earlier
        raise ValueError("lumi too low")
    elif (lumi <= 8.0): #run 1-2 added
        scale = sqrt((7.0/13.0*1.0 + 8.0/13.0*2.0)/(7.0/13.0*1.0 + 8.0/13.0*2.0 + (lumi-3)))
        return scale
    elif (lumi < 50.0): #run 3-4 added
        scale = sqrt((7.0/13.0*1.0 + 8.0/13.0*2.0)/(7.0/13.0*1.0 + 8.0/13.0*2.0 + 5.0 + (lumi-8.0)*14.0/13.0))
        return scale
    elif (lumi <= 300.0): #run 5 added
        scale = sqrt((7.0/13.0*1.0 + 8.0/13.0*2.0)/(7.0/13.0*1.0 + 8.0/13.0*2.0 + 5.0 + (lumi-8.0)*14.0/13.0))
        return scale
    else:
        raise

# numbers provided by Christoph Langenbruch
_lhcb_errors = np.array([
    sqrt((0.036)**2 + (0.017)**2),
    sqrt((0.038)**2 + (0.004)**2),
    sqrt((0.057)**2 + (0.004)**2),
    sqrt((0.050)**2 + (0.005)**2),
    sqrt((0.034)**2 + (0.007)**2),
    sqrt((0.050)**2 + (0.006)**2),
    sqrt((0.058)**2 + (0.008)**2),
    sqrt((0.042)**2 + (0.004)**2),
    sqrt((0.030)**2 + (0.008)**2),
    sqrt((0.033)**2 + (0.009)**2),
    sqrt((0.041)**2 + (0.007)**2),
    sqrt((0.037)**2 + (0.009)**2),
    sqrt((0.027)**2 + (0.009)**2),
    sqrt((0.043)**2 + (0.006)**2),
    sqrt((0.045)**2 + (0.003)**2),
    sqrt((0.039)**2 + (0.002)**2)
])

# low q^2 correlation
_lhcb_correlation_low = flavio.measurements._fix_correlation_matrix([[1.00, -0.04, 0.05, 0.03, 0.05, -0.04, -0.01, 0.08],
                                            [1.00, -0.05, -0.00, 0.05, 0.01, 0.01, -0.01],
                                            [1.00, -0.05, -0.11, -0.02, -0.01, 0.05],
                                            [1.00, -0.07, -0.01, -0.02, -0.04],
                                            [1.00, 0.02, -0.02, -0.04],
                                            [1.00, 0.04, -0.01],
                                            [1.00, -0.03],
                                            [1.00]], 8)
# high q^2 correlation
_lhcb_correlation_high = flavio.measurements._fix_correlation_matrix([[1.00, 0.17, -0.03, -0.02, -0.39, 0.01, -0.00, 0.11],
                                        [1.00, -0.15, -0.19, 0.05, -0.02, -0.04, -0.02],
                                        [1.00, 0.06, -0.12, 0.03, 0.14, 0.01],
                                        [1.00, -0.12, 0.12, 0.04, 0.02],
                                        [1.00, 0.00, -0.02, -0.01],
                                        [1.00, 0.24, -0.19],
                                        [1.00, -0.13],
                                        [1.00]], 8)


def lhcb_covariance(lumi):
    scale = lhcb_scale_uncertainty(lumi)
    err = scale * _lhcb_errors
    Z = np.zeros((8, 8))  # zero correlation between hi & lo q^2
    corr = np.block([[_lhcb_correlation_low, Z], [Z, _lhcb_correlation_high]])
    cov = np.outer(err, err) * corr
    return cov


covariances['LHCb']['Phase II'] = lhcb_covariance(lumi=300)


def C9C10scen(C9, C10):
    return Wilson({'C9_bsmumu': C9, 'C10_bsmumu': C10, }, 4.8, 'WET', 'flavio')


npscenarios['SM'] = C9C10scen(C9=0, C10=0)
npscenarios['Scenario I'] = C9C10scen(C9=-1.4, C10=0)
npscenarios['Scenario II'] = C9C10scen(C9=-0.7, C10=0.7)


def compute_sm_covariance(N=100, threads=4):
    """Compute the SM covariance for all observables."""
    # all observables from all experiments at all phases
    obs = list(set([o
                    for exp in observables.values()
                    for phase in exp.values()
                    for o in phase]))
    return flavio.sm_covariance(obs, N=N, threads=threads)


# Construct Likelihoods

likelihoods['LHCb']['present'] = FastLikelihood(
    'Likelihood LHCb present',
    observables=observables['LHCb']['present'],
    include_measurements=['LHCb B->K*mumu 2015 S 1.1-6',
                          'LHCb B->K*mumu 2015 S 15-19'],
)

np_measurement('Projection LHCb Phase II',
               w=npscenarios['Scenario I'],
               observables=observables['LHCb']['Phase II'],
               covariance=covariances['LHCb']['Phase II'])

likelihoods['LHCb']['Phase II'] = FastLikelihood(
    'Likelihood LHCb Phase II',
    observables=observables['LHCb']['Phase II'],
    include_measurements=['Projection LHCb Phase II'],
)

likelihoods['CMS']['present'] = FastLikelihood(
    'Likelihood CMS present',
    observables=observables['CMS']['present'],
    include_measurements=['CMS B->K*mumu 2017 P5p'],
)

np_measurement('Projection CMS Phase II',
               w=npscenarios['Scenario II'],
               observables=observables['CMS']['Phase II'],
               covariance=covariances['CMS']['Phase II'])

likelihoods['CMS']['Phase II'] = FastLikelihood(
    'Likelihood CMS Phase II',
    observables=observables['CMS']['Phase II'],
    include_measurements=['Projection CMS Phase II'],
)


if __name__ == '__main__':

    if os.path.exists(SMCOV):
        logging.info("Found existing covariance file; using that.")
        with open(SMCOV, 'rb') as f:
            smcov = pickle.load(f)
    else:
        logging.info("Computing covariance ...")
        smcov = compute_sm_covariance(N=N, threads=THREADS)

        logging.info("Done. Saving covariance.")
        with open(SMCOV, 'wb') as f:
            pickle.dump(smcov, f)

    par = flavio.default_parameters.get_central_all()

    for exp, vexp in likelihoods.items():
        for phase, L in vexp.items():
            smcov_dict = dict(covariance=smcov,
                              observables=L.observables)
            L.sm_covariance.load_dict(smcov_dict)
            L.make_measurement()

            def log_likelihood(x):
                w = C9C10scen(*x)
                return L.log_likelihood(par, w)

            logging.info("Computing plot {} {} ...".format(exp, phase))

            plotdata[exp][phase] = fpl.likelihood_contour_data(
                log_likelihood,
                XMIN, XMAX, YMIN, YMAX,
                steps=STEPS, threads=THREADS,
                n_sigma=(1, 2, 3, 4, 5))

    logging.info("Done! Saving all plots to file {}.".format(PDAT))
    with open(PDAT, 'wb') as f:
        pickle.dump(plotdata, f)
