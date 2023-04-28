
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
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
        # raise ValueError('The theshold is never crossed!')
        return ts[-1]
    return ts[time_index]


def expected_exit_time(Integrator, gamma, ic, T, log_time, ids, path, icid = '', different_initial_conditions=False, **kwargs):
    try:
        to_be_run_on = list(ids)
    except TypeError:
        to_be_run_on = range(ids)

    try:
        iter(ic)
        assert(len(ic)==len(to_be_run_on))
        assert(different_initial_conditions)
        d = ic[0].W0.shape[1]
        p = ic[0].W0.shape[0]
        k = ic[0].Wt.shape[0]
    except TypeError:
        assert(not different_initial_conditions)
        d = ic.W0.shape[1]
        p = ic.W0.shape[0]
        k = ic.Wt.shape[0]

    def extract_kwargs(kws):
        return {kw:kwargs[kw] for kw in kws if kw in kwargs}

    # run_integration = None
    # Simulation
    if issubclass(Integrator, BaseSimulation):
        def single_exit_time(id):
            intgr = Integrator(
                d, p, k, 
                Wt=ic[id].Wt,
                W0 = ic[id].W0,
                gamma=gamma,
                activation = kwargs['activation'],
                **extract_kwargs(['noise', 'disable_QM_save', 'extra_metrics']),
                seed = (0 if different_initial_conditions else id)
            )
            intgr_result = SimulationResult(
                initial_condition='time-measure'+(icid.format(icid=id) if different_initial_conditions else icid),
                id = (0 if different_initial_conditions else id)
            )
            intgr_result.from_file_or_run(
                intgr, 
                decades = log_time+np.log10(d*p/gamma),
                path=path,
                show_progress=False,
                **extract_kwargs(['force_run', 'force_read', 'save_per_decade'])
            )
            return exit_time(
                np.array(intgr_result.steps) * gamma/(p*d),
                np.array(intgr_result.risks),
                T
            )
    # SDE
    elif issubclass(Integrator, BaseSDE):
        def single_exit_time(id):
            intgr = Integrator(
                ic[id].Q, ic[id].M, d,
                dt = kwargs['dt'],
                **extract_kwargs(['noise_term','noise', 'gamma_over_p', 'quadratic_terms']),
                seed = (0 if different_initial_conditions else id)
            )
            intgr_result = PhaseRetrievalSDEResult(
                initial_condition='time-measure'+(icid.format(icid=id) if different_initial_conditions else icid),
                id = (0 if different_initial_conditions else id)
            )
            intgr_result.from_file_or_run(
                intgr, 
                decades = log_time,
                path=path,
                show_progress=False,
                **extract_kwargs(['force_run', 'force_read', 'save_per_decade'])
            )
            return exit_time(
                np.array(intgr_result.times),
                np.array(intgr_result.risks),
                T
            )
    else:
        raise TypeError(f'Not recognized class type: {type(Integrator)}')

    # if kwargs['force_read']:
    #     pool = Pool(cpu=12)
    #     exit_times = np.array(
    #         list(pool.map(single_exit_time, to_be_run_on))
    #     )
    # else:
    exit_times = np.array(
        list(map(single_exit_time, to_be_run_on))
    )
    return exit_times.mean(), exit_times.std()
