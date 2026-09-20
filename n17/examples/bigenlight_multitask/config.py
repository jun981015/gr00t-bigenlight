"""UR7e, two RGB views, absolute joint commands; identical semantics to N1.5."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    EmbodimentTag,
    ModalityConfig,
)


CONFIG = {
    "video": ModalityConfig([0], ["scene", "wrist"]),
    "state": ModalityConfig([0], ["arm", "gripper"]),
    "action": ModalityConfig(
        list(range(16)),
        ["arm", "gripper"],
        action_configs=[
            ActionConfig(ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT),
            ActionConfig(ActionRepresentation.ABSOLUTE, ActionType.NON_EEF, ActionFormat.DEFAULT),
        ],
    ),
    "language": ModalityConfig([0], ["annotation.human.task_description"]),
}

register_modality_config(CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
