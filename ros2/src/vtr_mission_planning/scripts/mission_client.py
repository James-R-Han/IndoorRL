#!/usr/bin/env python

import uuid
from enum import Enum
import time
from vtr_mission_planning.ros_manager import RosManager
# Todo separate file
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from collections import OrderedDict

from unique_identifier_msgs.msg import UUID

from vtr_messages.action import Mission
from vtr_messages.srv import MissionPause


class MissionClient(RosManager):

  class Notification(Enum):
    """Enumerates possible notifications that might come back from ROS;
    overloads parent definition
    """
    Feedback = 0
    Complete = 1
    Cancel = 2
    Error = 3
    Started = 4
    NewGoal = 5
    StatusChange = 6
    RobotChange = 7
    PathChange = 8
    GraphChange = 9
    SafetyStatus = 10
    OverlayRefresh = 11

  def __init__(self):
    super().__init__()

  @RosManager.on_ros
  def setup_ros(self, *args, **kwargs):
    """Sets up necessary ROS communications"""
    # Mission action server
    self._action = ActionClient(self, Mission, "manager")
    self._action.wait_for_server()
    self._goals = OrderedDict()
    self._feedback = {}
    #
    self._pause = self.create_client(MissionPause, 'pause')
    self._pause.wait_for_service()

  @RosManager.on_ros
  def set_pause(self, pause=True):
    """Sets the pause state of the mission server

    :param paused: whether or not to pause the server
    """
    req = MissionPause.Request()
    req.pause = pause
    res = self._pause.call(req)
    # print("Pause result:", str(res.response_code))
    # return res.response_code

  @RosManager.on_ros
  def add_goal(self,
               goal_type=None,
               path=(),
               pause_before=0,
               pause_after=0,
               vertex=2**64 - 1):
    """Adds a new goal inside the ROS process

    :param goal_type enum representing the type of goal to add
    :param path list of vertices to visit
    :param pause_before duration in seconds to pause before execution
    :param pause_after duration in seconds to pause after execution:
    """

    if goal_type not in [
        Mission.Goal.IDLE,
        Mission.Goal.TEACH,
        Mission.Goal.REPEAT,
        Mission.Goal.MERGE,
        Mission.Goal.LOCALIZE,
        Mission.Goal.OTHER,
    ]:
      raise RuntimeError("Goal of type %d not in range [0,5]" % (goal_type,))

    goal = Mission.Goal()
    goal.target = int(goal_type)
    goal.path = path
    goal.vertex = vertex
    # goal.pause_before = rclpy.duration.Duration(seconds=pause_before)
    # goal.pause_after = rclpy.duration.Duration(seconds=pause_after)
    goal_uuid = UUID(uuid=list(uuid.uuid4().bytes))
    goal_uuid_str = goal_uuid.uuid.tostring()

    self.get_logger().info(
        "Add a goal with id <{}> of type {}, path {}, pause before {}, pause after {}, vertex {}."
        .format(goal_uuid_str, goal_type, path, pause_before, pause_after,
                vertex))

    send_goal_future = self._action.send_goal_async(
        goal, feedback_callback=self.feedback_callback, goal_uuid=goal_uuid)
    send_goal_future.add_done_callback(self.response_callback)

    return goal_uuid_str

  @RosManager.on_ros
  def cancel_all(self):
    """Cancels a goal inside the ROS process

    :param goal_id: goal id to be cancelled
    """
    # This is safe, because the callbacks that modify _queue block until this function returns and _lock is released
    for v in reversed(self._goals.values()):
      v.cancel_goal_async()

    return True

  @RosManager.on_ros
  def cancel_goal(self, uuid):
    """Cancels a goal inside the ROS process

    :param goal_id: goal id to be cancelled
    """
    # Check to make sure we are still tracking this goal
    if uuid not in self._goals.keys():
      return False

    # Cancel the goal
    self.get_logger().info("Cancel goal with id <{}>".format(uuid))
    self._goals[uuid].cancel_goal_async()
    return True

  @RosManager.on_ros
  def feedback_callback(self, feedback_handle):
    feedback = feedback_handle.feedback
    uuid = feedback_handle.goal_id.uuid.tostring()

    if not uuid in self._goals.keys():
      return

    if not uuid in self._feedback.keys():
      self._feedback[uuid] = feedback
      self.notify(self.Notification.Feedback, uuid, feedback)
      self.get_logger().info(
          "Goal with id <{}> gets first feedback saying percent complete {} and waiting {}"
          .format(uuid, feedback.percent_complete, feedback.waiting))
    else:
      old = self._feedback[uuid]
      self._feedback[uuid] = feedback

      if old.percent_complete != feedback.percent_complete or old.waiting != feedback.waiting:
        self.notify(self.Notification.Feedback, uuid, feedback)
      self.get_logger().info(
          "Goal with id <{}> gets updated feedback saying percent complete {} and waiting {}"
          .format(uuid, feedback.percent_complete, feedback.waiting))

  @RosManager.on_ros
  def response_callback(self, future):
    goal_handle = future.result()
    uuid = goal_handle.goal_id.uuid.tostring()
    if not goal_handle.accepted:
      self.get_logger().info(
          'Goal with id <{}> has been rejected.'.format(uuid))
      return
    self.get_logger().info('Goal with id <{}> has been accepted.'.format(uuid))
    self._goals[uuid] = goal_handle
    get_result_future = goal_handle.get_result_async()
    get_result_future.add_done_callback(
        lambda future, uuid=uuid: self.get_result_callback(uuid, future))

  @RosManager.on_ros
  def get_result_callback(self, uuid, future):
    result = future.result().result
    status = future.result().status

    if not uuid in self._goals.keys():
      return

    # Goal has completed successfully
    if status == GoalStatus.STATUS_SUCCEEDED:
      self.get_logger().info(
          'Goal with id <{}> succeeded with result: {}'.format(uuid, result))
      self.notify(self.Notification.Complete, uuid)
    # Goal has failed for an internal reason
    elif status == GoalStatus.STATUS_ABORTED:
      self.get_logger().info("Goal with id <{}> aborted.".format(uuid))
      self.notify(self.Notification.Error, uuid, status)
    # Goal was cancelled by the user
    elif status == GoalStatus.STATUS_CANCELED:
      self.get_logger().info("Goal with id <{}> cancelled.".format(uuid))
      self.notify(self.Notification.Cancel, uuid)
    # Goal has been started
    else:
      # GoalStatus.STATUS_EXECUTING -> will we ever enter this state?
      self.get_logger().info("Goal with id <{}> in unknown state.".format(uuid))
      self.notify(self.Notification.Started, uuid)

    # If the goal is in a terminal state, remove it from being tracked
    if status in [
        GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED,
        GoalStatus.STATUS_CANCELED
    ]:
      self.get_logger().info(
          "Goal with id <{}> deleted from queue.".format(uuid))
      del self._goals[uuid]
      if uuid in self._feedback.keys():
        del self._feedback[uuid]


if __name__ == "__main__":
  mc = MissionClient()
  time.sleep(0.1)
  # mc.set_pause(False)
  uuid = mc.add_goal(Mission.Goal.IDLE)
  time.sleep(0.1)
  mc.cancel_goal(uuid)
  uuid = mc.add_goal(Mission.Goal.TEACH)
  uuid = mc.add_goal(Mission.Goal.OTHER)
  uuid = mc.add_goal(Mission.Goal.IDLE)
  uuid = mc.add_goal(Mission.Goal.IDLE)
  # mc.cancel_all()
  # mc.shutdown()