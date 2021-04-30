#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import numpy as np

STATE_COUNT_THRESHOLD = 2
TL_LOOKAHEAD_WPS = 100 # Traffic light dectector will only be activated within the look ahead distance to reduce simulator lagging
LIGHT_STATE_MAP = {0:'Red', 1:'Yellow', 2: 'Green'}


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        # self.img_count = 851

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
        self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        # # Code to collect camera images for training
        # car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
        # if self.has_image and (light_wp > car_wp_idx + TL_LOOKAHEAD_WPS):
        #     # self.save_image(state)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if ((state == TrafficLight.RED) or (state == TrafficLight.Yellow)) else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

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


    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #Get classification
        state = self.light_classifier.get_classification(cv_image)
        if state==TrafficLight.UNKNOWN:
            rospy.loginfo("Unknown traffic light state")
            return state
        state_msg = LIGHT_STATE_MAP[state]
        rospy.loginfo("Traffic light ahead visible, detected state: %s", state_msg)
        return state


    def save_image(self, state):
        """Collect training dataset for traffic light dectoctor
        """
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        self.img_count +=1
        image_name = 'img_'+ str(self.img_count) +'.png'
        path = "/home/kaoozhinux16/workspace/CarND-Capstone/imgs/classifier/"
        image_path = path + "train/" + image_name
        cv2.imwrite(image_path, cv_image)
        file_path = path + "label.txt"
        label_file = open(file_path,"a")
        label_file.write(str(self.img_count) + "," + str(state)+"\n")
        label_file.close() #to change file access modes

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        light_wp = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        #TODO find the closest visible traffic light (if one exists)
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            # diff_min = len(self.waypoints.waypoints)
            diff_min = TL_LOOKAHEAD_WPS 
            for i, stop_line_position in enumerate(stop_line_positions): 
                temp_idx = self.get_closest_waypoint(stop_line_position[0], stop_line_position[1])
                diff = temp_idx - car_wp_idx
                if 0 <= diff < diff_min:
                    diff_min = diff
                    closest_light = self.lights[i]
                    light_wp = temp_idx

        if closest_light:
            state = self.get_light_state(closest_light)
            return light_wp, state
        rospy.loginfo("No visible Traffic light ahead")
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

    

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
