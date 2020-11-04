#! /usr/bin/python

import rospy
from geometry_msgs.msg import Twist
from rospy import Publisher, Subscriber
from turtlesim.msg import Pose
import math
import numpy as np


class Hitman:
    def __init__(self, eps=1e-2, hitman_speed=1.5):
        self.victim_sub = Subscriber('/turtle1/pose', Pose, self.follow)
        self.hitman_sub = Subscriber('/hitman/pose', Pose, self.update)
        self.hitman_pub = Publisher('/hitman/cmd_vel', Twist, queue_size=10)
        self.victim_pose = Pose()
        self.eps = eps
        self.hitman_speed = hitman_speed

    def update(self, victim_pose):
        self.victim_pose = victim_pose

    def follow(self, hitman_pose):
        victim_coordinates = np.array([self.victim_pose.x, self.victim_pose.y])
        hitman_coordinates = np.array([hitman_pose.x, hitman_pose.y])

        # calculate distance from hitman to victim
        distance = np.linalg.norm(hitman_coordinates - victim_coordinates)
        if distance <= self.eps:
            return

        # calculate desired angle
        angle = math.atan2(hitman_pose.y - self.victim_pose.y, hitman_pose.x - self.victim_pose.x) - self.victim_pose.theta
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi

        msg = Twist()
        msg.linear.x = min(distance, self.hitman_speed)
        msg.angular.z = angle
        self.hitman_pub.publish(msg)


rospy.init_node('main')
hitman_speed = float(rospy.get_param("/hitman_speed"), 1.5)
Hitman(hitman_speed=hitman_speed)
rospy.spin()
