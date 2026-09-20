"""UR7e + Robotiq 2F-85: measured joints/state, commanded joints/actions."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


carrot_config = {
    "video": ModalityConfig([0], ["scene", "wrist"]),
    "state": ModalityConfig([0], ["arm", "gripper"]),
    "action": ModalityConfig(
        list(range(16)),
        ["arm", "gripper"],
        action_configs=[
            # Disk actions remain absolute radians. GR00T converts each chunk
            # relative to the current measured arm state, then normalizes it.
            ActionConfig(ActionRepresentation.RELATIVE, ActionType.NON_EEF, ActionFormat.DEFAULT),
            # Preserve the recorded convention: 0=open, 1=closed.
            ActionConfig(ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig([0], ["annotation.human.task_description"]),
}

register_modality_config(carrot_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
