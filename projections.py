"""WC Projections for HL LHC & Belle-II"""

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

N = 2000
THREADS = 8
STEPS = 20

XMIN = -2
XMAX = 1
YMIN = -1.5
YMAX = 1.5


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


observables = {}
covariances = {}
likelihoods = {}
plotdata = {}
npscenarios = {}

# CMS

observables['CMS present'] = [
    ('<P5p>(B0->K*mumu)', 1, 2),
    ('<P5p>(B0->K*mumu)', 2, 4.3),
    ('<P5p>(B0->K*mumu)', 4.3, 6),
    'BR(Bs->mumu)'
]


observables['CMS Phase II'] = observables['CMS present']
observables['CMS Phase I'] = observables['CMS Phase II']
observables['ATLAS Phase I'] = observables['CMS Phase I']
observables['ATLAS Phase II'] = observables['CMS Phase II']

# Building the CMS Phase II covariance matrix

# numbers provided by Sandra Malvezzi
# columns: obs, rows: unc. sources
_cms_uncorr_sys_3000 = np.array([
    [0.0115, 0.0065, 0.0075, 0],
    [0.00177, 0.000417, 0.00497, 0],
    [0.0006, 0.0012, 0.0016, 0],
    [0.0014, 0.0076, 0.0108, 0],
    [0, 0, 0, 0],
])
_cms_uncorr_sys_300 = np.array([
    [0.0115, 0.0065, 0.0075, 0],
    [0.0056, 0.0013, 0.0157, 0],
    [0.0014, 0.0032, 0.004, 0],
    [0.0035, 0.0195, 0.0272, 0],
    [0, 0, 0, 0],
])
_cms_corr_sys_3000 = np.array([
    [0.0025, 0.03, 0.0325, 0],
    [0.0065, 0.0075, 0.007, 0],
    [0.0095, 0.0095, 0.0095, 0],
    [5.25e-05, 0.0005, 0.0007, 0],
])
_cms_corr_sys_300 = np.array([
    [0.0025, 0.03, 0.0325, 0],
    [0.0065, 0.0075, 0.007, 0],
    [0.0095, 0.0095, 0.0095, 0],
    [0.0001, 0.0005, 0.0007, 0],
])
_cms_stat_300 = np.array([0.045, 0.045, 0.029, 0.12 * 3.5e-9])
_cms_stat_3000 = np.array([0.014, 0.014, 0.009, 0.11 * 3.5e-9])

covariances['CMS Phase I stat'] = _cms_stat_300**2 * np.eye(4)  # uncorrelated stat
covariances['CMS Phase II stat'] = _cms_stat_3000**2 * np.eye(4)  # uncorrelated stat

_var_corr = np.sum(_cms_corr_sys_3000**2, axis=0)  # error^2 of fully correlated systematic
_corr = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 1]])  # correlation matrix
covariances['CMS Phase II stat+sys'] = (
covariances['CMS Phase II stat']
+ np.sum(_cms_uncorr_sys_3000**2, axis=0) * np.eye(4)  # uncorrelated sys
+ np.outer(_var_corr, _var_corr) * _corr   # fully correlated sys
)

_var_corr = np.sum(_cms_corr_sys_300**2, axis=0)  # error^2 of fully correlated systematic
covariances['CMS Phase I stat+sys'] = (
covariances['CMS Phase I stat']
+ np.sum(_cms_uncorr_sys_300**2, axis=0) * np.eye(4)  # uncorrelated sys
+ np.outer(_var_corr, _var_corr) * _corr   # fully correlated sys
)

#  ATLAS = CMS

covariances['ATLAS Phase I stat'] = covariances['CMS Phase I stat']
covariances['ATLAS Phase II stat'] = covariances['CMS Phase II stat']
covariances['ATLAS Phase I stat+sys'] = covariances['CMS Phase I stat+sys']
covariances['ATLAS Phase II stat+sys'] = covariances['CMS Phase II stat+sys']

# LHCb

observables['LHCb present'] = [
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
    'BR(Bs->mumu)'
]

observables['LHCb Phase II'] = observables['LHCb present']
observables['LHCb Phase I'] = observables['LHCb Phase II']

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
_lhcb_err_stat = np.array([0.036, 0.038, 0.057, 0.050, 0.034, 0.050, 0.058, 0.042, 0.030, 0.033, 0.041, 0.037, 0.027, 0.043, 0.045, 0.039])
_lhcb_err_sys = np.array([0.017, 0.004, 0.004, 0.005, 0.007, 0.006, 0.008, 0.004, 0.008, 0.009, 0.007, 0.009, 0.009, 0.006, 0.003, 0.002])

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


def lhcb_covariance_ksmumu(lumi, include_sys=True):
    scale = lhcb_scale_uncertainty(lumi)
    if include_sys:
        err = scale * np.sqrt(_lhcb_err_stat**2 + _lhcb_err_sys**2)
    else:
        err = scale * _lhcb_err_stat
    Z = np.zeros((8, 8))  # zero correlation between hi & lo q^2
    corr = np.block([[_lhcb_correlation_low, Z], [Z, _lhcb_correlation_high]])
    cov = np.outer(err, err) * corr
    return cov


def lhcb_covariance(lumi, include_sys=True):
    cov_ksmumu = lhcb_covariance_ksmumu(lumi, include_sys=include_sys)
    D = len(cov_ksmumu)
    cov = np.zeros((D + 1, D + 1))
    cov[:D, :D] = cov_ksmumu
    # BR(Bs->mumu)
    if lumi == 23:
        cov[D, D] = (0.30e-9)**2
    elif lumi == 300:
        cov[D, D] = (0.16e-9)**2
    else:
        raise ValueError()
    return cov


covariances['LHCb Phase I stat'] = lhcb_covariance(lumi=23, include_sys=False)
covariances['LHCb Phase I stat+sys'] = lhcb_covariance(lumi=23, include_sys=True)
covariances['LHCb Phase II stat'] = lhcb_covariance(lumi=300, include_sys=False)
covariances['LHCb Phase II stat+sys'] = lhcb_covariance(lumi=300, include_sys=True)


def C9C10scen(C9, C10):
    return Wilson({'C9_bsmumu': C9, 'C10_bsmumu': C10, }, 4.8, 'WET', 'flavio')


npscenarios['SM'] = C9C10scen(C9=0, C10=0)
npscenarios['Scenario I'] = C9C10scen(C9=-1.4, C10=0)
npscenarios['Scenario II'] = C9C10scen(C9=-0.7, C10=0.7)


allobs = observables['CMS present'][:-1] + observables['LHCb present']

def compute_sm_covariance(N=100, threads=4):
    """Compute the SM covariance for all observables."""
    # all observables from all experiments at all phases
    return flavio.sm_covariance(allobs, N=N, threads=threads)


# Construct Likelihoods

# future

for phase in ['Phase I', 'Phase II', ]:
    for errscen in ['stat', 'stat+sys']:
        for npscen in ['Scenario I', 'Scenario II', 'SM']:
            for exp in ['LHCb', 'CMS', 'ATLAS']:
                np_measurement('Projection {} {} {} {}'.format(exp, phase, errscen, npscen),
                               w=npscenarios[npscen],
                               observables=observables[' '.join([exp, phase])],
                               covariance=covariances[' '.join([exp, phase, errscen])])

            likelihoods[' '.join([phase, errscen, npscen])] = FastLikelihood(
                'Likelihood LHC {} {} {}'.format(phase, errscen, npscen),
                observables=list(set(
                                       observables['LHCb {}'.format(phase)]
                                     + observables['CMS {}'.format(phase)]
                                     + observables['ATLAS {}'.format(phase)]
                                     )),
                include_measurements=[
                    'Projection LHCb {} {} {}'.format(phase, errscen, npscen),
                    'Projection CMS {} {} {}'.format(phase, errscen, npscen),
                    'Projection ATLAS {} {} {}'.format(phase, errscen, npscen),
                    ]
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

    for name, L in likelihoods.items():
            smcov_dict = dict(covariance=smcov,
                              observables=allobs)
            L.sm_covariance.load_dict(smcov_dict)
            L.make_measurement()

            def log_likelihood(x):
                w = C9C10scen(*x)
                return L.log_likelihood(par, w)

            logging.info("Computing plot {} ...".format(name))

            plotdata[name] = fpl.likelihood_contour_data(
                log_likelihood,
                XMIN, XMAX, YMIN, YMAX,
                steps=STEPS, threads=THREADS,
                n_sigma=(1, 2, 3, 4, 5))

    logging.info("Done! Saving all plots to file {}.".format(PDAT))
    with open(PDAT, 'wb') as f:
        pickle.dump(plotdata, f)
