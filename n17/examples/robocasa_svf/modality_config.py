"""Use N1.7's Panda Omron schema, with an explicitly executed 16-action RL chunk."""

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag


tag = EmbodimentTag.ROBOCASA_PANDA_OMRON.value
config = deepcopy(MODALITY_CONFIGS[tag])
config["action"].delta_indices = list(range(16))
MODALITY_CONFIGS[tag] = config
