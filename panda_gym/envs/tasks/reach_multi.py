from typing import Any, Dict, Union, List

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance
from random import sample


class ReachMulti(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30)

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_cylinder(
            body_name="target_cylinder",
            radius=0.02,
            height=1,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        self.sim.create_sphere(
            body_name="target_sphere",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        if "sphere" in self.task:
            task_id = np.array([0])
        elif "cylinder" in self.task:
            task_id = np.array([1])
        else:
            task_id = np.array([])
        return task_id  # no task-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        if "cylinder" in self.task:
            ee_position[2] = 0 # kind of acc
        return ee_position

    def reset(self) -> None:
        self.task = self._sample_task() #todo
        self.goal = self._sample_goal()
        if "cylinder" in self.task:
            self.sim.set_base_pose("target_cylinder", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("target_sphere", np.array([-0.6, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
        elif "sphere" in self.task:
            self.sim.set_base_pose("target_cylinder", np.array([-0.6, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
            self.sim.set_base_pose("target_sphere", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_task(self) -> List[str]:
        """Randomize task."""
        task_list = ["sphere", "cylinder"]
        task = sample(task_list, 1)
        return task

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if "cylinder" in self.task:
            goal[2] = 0 # kind of accounts for the cylinder
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=np.float64)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return -d
