from .reservoir_random import rsvrRandom
from .mlab_stratified_reservoir import PRS_mlab as PRS_mlab

reservoir = {
    'random': rsvrRandom,
    'prs_mlab': PRS_mlab
}
