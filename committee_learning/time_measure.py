
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
        raise ValueError('The theshold is never crossed!')
        # return ts[-1]
    return ts[time_index]

def escape_time(ts, As, factor, A0 = None):
    assert(ts.shape == As.shape)
    if A0 == None:
        A0 = As[0]
    time_index = np.argmin(As < A0*factor)

    if As[time_index] < A0*factor:
        raise ValueError('The theshold is never crossed!')
        # return ts[-1]
    return ts[time_index]


def expected_exit_time(Integrator, gamma, ic, T, log_time, ids, path, icid = '', different_initial_conditions=False, xor_seed = 0, allow_missing=False, **kwargs):
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

    # Simulation
    if issubclass(Integrator, BaseSimulation):
        def single_exit_time(id):
            intgr = Integrator(
                d, p, k, 
                Wt=ic[id].Wt if different_initial_conditions else ic.Wt,
                W0 = ic[id].W0 if different_initial_conditions else ic.W0,
                gamma=gamma,
                activation = kwargs['activation'],
                **extract_kwargs(['noise', 'disable_QM_save', 'extra_metrics']),
                # seed = 0
                seed = (id^xor_seed)
            )
            intgr_result = SimulationResult(
                initial_condition='time-measure'+(icid.format(icid=id) if different_initial_conditions else icid),
                id = (0 if different_initial_conditions else id)
                # id = id
            )
            intgr_result.from_file_or_run(
                intgr, 
                decades = log_time+np.log10(d*p/gamma),
                path=path,
                show_progress=False,
                **extract_kwargs(['force_run', 'force_read', 'save_per_decade'])
            )
            return exit_time(
                np.array(intgr_result.steps)*gamma/(p*d),
                np.array(intgr_result.risks),
                T
            )
    # SDE
    elif issubclass(Integrator, BaseSDE):
        def single_exit_time(id):
            if p > 0:
                intgr = Integrator(
                    ic[id].Q if different_initial_conditions else ic.Q,
                    ic[id].M if different_initial_conditions else ic.M,
                    d,
                    dt = kwargs['dt'],
                    **extract_kwargs(['noise_term','noise', 'gamma_over_p', 'quadratic_terms']),
                    seed = (id^xor_seed)
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
        raise TypeError(f'Not recognized class type: {Integrator}')

    exit_times_results = []
    for id in to_be_run_on:
        try:
            exit_times_results.append(single_exit_time(id))
        except FileNotFoundError as e:
            if allow_missing:
                pass
            else:
                raise e
    exit_times = np.array(exit_times_results)
    return exit_times.mean(), exit_times.std()
