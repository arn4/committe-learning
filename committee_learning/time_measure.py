
import numpy as np
import multiprocessing
from tqdm import tqdm

from .simulation.base import BaseSimulation
from .sde.base import BaseSDE
from .result import SimulationResult, PhaseRetrievalSDEResult

def exit_time(ts, Rs, T, R0 = None):
    assert(ts.shape == Rs.shape)
    if R0 == None:
        R0 = Rs[0]
    time_index = np.argmin(Rs > R0*(1.-T))

    if Rs[time_index] > R0*(1.-T):
        raise ValueError('The theshold is never crossed!')
    return ts[time_index]


def expected_exit_time(Integrator, gamma, ic, T, log_time, ids, path, **kwargs):
    d = ic.W0.shape[1]
    p = ic.W0.shape[0]
    k = ic.Wt.shape[0]
    try:
        to_be_run_on = list(ids)
    except TypeError:
        to_be_run_on = range(ids)

    def extract_kwargs(kws):
        return {kw:kwargs[kw] for kw in kws if kw in kwargs}

    run_integration = None
    # Simulation
    if issubclass(Integrator, BaseSimulation):
        def ri(id):
            intgr = Integrator(
                d, p, k, 
                Wt=ic.Wt,
                W0 = ic.W0,
                gamma=gamma,
                activation = kwargs['activation']
                **extract_kwargs(['noise', 'disable_QM_save', 'extra_metrics'])
            )
            intgr_result = SimulationResult(
                initial_condition='time-measure',
                id = id
            )
            intgr_result.from_file_or_run(
                intgr, 
                decades = log_time+np.log10(d*p/gamma),
                path=path,
                show_progress=False,
                **extract_kwargs(['force_run', 'force_read', 'save_per_decade'])
            )
            return (
                np.array(intgr_result.steps),
                np.array(intgr_result.risks)
            )
        run_integration = ri
    # SDE
    elif issubclass(Integrator, BaseSDE):
        def ri(id):
            intgr = Integrator(
                ic.Q, ic.M, d,
                dt = kwargs['dt'],
                **extract_kwargs(['noise_term','noise', 'gamma_over_p', 'quadratic_terms']),
                seed = id
            )
            intgr_result = PhaseRetrievalSDEResult(
                initial_condition='time-measure',
                id = id
            )
            intgr_result.from_file_or_run(
                intgr, 
                decades = log_time,
                path=path,
                show_progress=False,
                **extract_kwargs(['force_run', 'force_read', 'save_per_decade'])
            )
            return (
                np.array(intgr_result.times),
                np.array(intgr_result.risks)
            )
        run_integration = ri
    else:
        raise TypeError(f'Not recognized class type: {type(Integrator)}')

    exit_times = np.array(
        [exit_time(*run_integration(id), T) for id in tqdm(to_be_run_on)]
    )
    return exit_times.mean(), exit_times.std()