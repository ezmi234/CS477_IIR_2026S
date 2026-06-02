from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class PickPlaceTask:
    object_name: str
    destination: str
    source_command: str = ''
    attempt: int = 0

    def label(self):
        return f'{self.object_name} -> {self.destination}'


class ExecutorState(Enum):
    STANDBY = auto()
    RECEIVE_COMMAND = auto()
    PARSE_COMMAND = auto()
    BUILD_TASK_QUEUE = auto()
    CHECK_TASK_QUEUE = auto()
    NEXT_TASK = auto()
    GET_OBJECT_POSE = auto()
    COMPUTE_GRASP = auto()
    PICK = auto()
    PLACE = auto()
    DONE = auto()
