#!/usr/bin/env python3
"""Diagnostic selector for the RACER/PPO task handshake; not a flight policy."""

import rospy

from exploration_manager.msg import RLTask, RLTarget, RLViewpointSelection


class TaskTestSelector:
    def __init__(self) -> None:
        self.drone_id = int(rospy.get_param("~drone_id", 1))
        self.force_index = int(rospy.get_param("~candidate_index", -1))
        self.last_task = None
        suffix = str(self.drone_id)
        self.selection_pub = rospy.Publisher(
            f"/rl_navigation/viewpoint_selection_{suffix}",
            RLViewpointSelection,
            queue_size=1,
        )
        rospy.Subscriber(
            f"/rl_navigation/task_{suffix}", RLTask, self.task_callback, queue_size=1
        )
        rospy.Subscriber(
            f"/rl_navigation/target_{suffix}", RLTarget, self.target_callback, queue_size=1
        )

    def task_callback(self, task: RLTask) -> None:
        if task.drone_id != self.drone_id or not task.candidate_positions:
            return
        if self.last_task == task.task_id:
            return
        self.last_task = task.task_id
        if self.force_index >= 0:
            index = self.force_index
        elif task.candidate_visible_voxels:
            index = max(
                range(len(task.candidate_visible_voxels)),
                key=task.candidate_visible_voxels.__getitem__,
            )
        else:
            index = 0
        selection = RLViewpointSelection()
        selection.header.stamp = rospy.Time.now()
        selection.header.frame_id = task.header.frame_id
        selection.task_id = task.task_id
        selection.drone_id = self.drone_id
        selection.candidate_index = index
        self.selection_pub.publish(selection)
        rospy.loginfo(
            "test selector sent task=%d candidate=%d/%d",
            task.task_id,
            index,
            len(task.candidate_positions),
        )

    def target_callback(self, target: RLTarget) -> None:
        rospy.loginfo_throttle(
            1.0,
            "target task=%d active=%s xyz=(%.2f, %.2f, %.2f)",
            target.target_id,
            target.active,
            target.position.x,
            target.position.y,
            target.position.z,
        )


if __name__ == "__main__":
    rospy.init_node("rl_task_test_selector")
    TaskTestSelector()
    rospy.spin()
