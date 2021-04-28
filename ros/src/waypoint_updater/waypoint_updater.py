#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math

MAX_DECEL = .5
'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 70 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.red_light_wp = -1

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.loginfo("I will publish to the topic %s", topic)
        # rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)


        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below


        self.loop()

        # rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        # pass
        self.pose = msg
    
    def loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                # closet_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints()
            rate.sleep()
    
    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoints_tree.query([x,y],1)[1]

        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        # Check if closest coord is in front of the car
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect - prev_vect, pos_vect-cl_vect)
        if val>0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)


        return closest_idx
    
    def publish_waypoints(self):
        lane = self.create_lane()
        # if self.red_light_wp <= closest_idx + LOOKAHEAD_WPS:

        # closest_wp_vel = self.get_waypoint_velocity(self.base_waypoints.waypoints[closest_idx])

        # for i in range(LOOKAHEAD_WPS):

        #     target_vel = closest_wp_vel + 
        #     self.set_waypoint_velocity(self, lane.waypoints, i, velocity)
        # rospy.loginfo("red_light_wp" + str(self.red_light_wp))

        self.final_waypoints_pub.publish(lane)

    def create_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        waypoints = self.base_waypoints.waypoints[closest_idx: closest_idx + LOOKAHEAD_WPS]
        
        # Generate path to decelerate to stop if red traffic lane detected ahead within the lookahead distance
        if self.red_light_wp ==-1 or (self.red_light_wp >= closest_idx + LOOKAHEAD_WPS):
            lane.waypoints = waypoints
        # Traffic light ahead need to generate deceleration velocity profile
        else:
            lane.waypoints = self.decelerate_wp(waypoints, closest_idx)
            # lane.waypoints = waypoints
        
        # print("red_wp" + str(self.red_light_wp))
            
        return lane

    def decelerate_wp(self, waypoints, closest_idx):
        
        # end_idx = max(self.red_light_wp - closest_idx -2, 0)
        # vel = 0.0
        # final_wps = waypoints
        # base_vel = final_wps[end_idx].twist.twist.linear.x
        # i = end_idx-1
        
        # while vel<=base_vel and i >=0:
        #     end_idx = max(self.red_light_wp - closest_idx -2, 0)
        #     dist = self.distance(waypoints,i,end_idx)
        #     base_vel = waypoints[i].twist.twist.linear.x
        #     vel = math.sqrt(2*MAX_DECEL*dist)
        #     if vel<1.:
        #         vel = 0.
            
        #     final_wps[i].twist.twist.linear.x = vel
        #     i-=1

        # print("first wp to decelerate" + str(i))

        final_wps =[]
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            end_idx = max(self.red_light_wp - closest_idx -2, 0)
            dist = self.distance(waypoints,i,end_idx)
            vel = math.sqrt(2*MAX_DECEL*dist)
            if vel<1.:
                vel = 0.
            
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            final_wps.append(p)

        return final_wps


    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        # pass
        self.red_light_wp = msg.data
        # print(self.red_light_wp)


    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
