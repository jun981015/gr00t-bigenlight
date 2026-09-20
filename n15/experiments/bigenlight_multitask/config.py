"""N1.5 UR7e mapping; do not interpret joint positions as end-effector deltas."""

from typing import ClassVar

from gr00t.experiment.data_config import So100DataConfig


class UR7eDataConfig(So100DataConfig):
    video_keys: ClassVar[list[str]] = ["video.scene", "video.wrist"]
    state_keys: ClassVar[list[str]] = ["state.arm", "state.gripper"]
    action_keys: ClassVar[list[str]] = ["action.arm", "action.gripper"]
    language_keys: ClassVar[list[str]] = ["annotation.human.task_description"]
    observation_indices: ClassVar[list[int]] = [0]
    action_indices: ClassVar[list[int]] = list(range(16))
