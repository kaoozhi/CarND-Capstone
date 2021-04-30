from styx_msgs.msg import TrafficLight
import numpy as np
import os
import cv2
import tensorflow as tf
import rospy

CLASS_MAP = {0: TrafficLight.RED, 1:TrafficLight.GREEN, 2:TrafficLight.YELLOW}

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # Set path:
        basepath = os.path.dirname(os.path.abspath(__file__))
        # path_to_model = os.path.join(basepath, 'ssd_mobilenet_v1_300')
        # path_to_model = os.path.join(basepath, 'ssd_mobilenet_v1_240')
        # path_to_model = os.path.join(basepath, 'ssd_inception_v2_512')
        #  path_to_model = os.path.join(basepath, 'ssd_inception_v1_300')
        path_to_model = os.path.join(basepath, 'faster_rcnn_inception')
        path_to_graph = os.path.join(path_to_model, 'frozen_inference_graph.pb')
        path_to_labels = os.path.join(path_to_model, 'label_map.pbtxt')
        
        self.detection_graph = self.load_graph(path_to_graph)
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.detection_graph, config=self.tf_config)
        # Run artificial random image to warm up TensorFlow's memory allocator
        warmup_iter = 10
        for iter in range(warmup_iter):
            synth_data = np.random.randint(low=0, high=255, size=(600, 800, 3), dtype=np.uint8)
            self.inference(synth_data)
        rospy.loginfo("Traffic light detector initialized")
            
    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def inference(self, image):
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        image_np_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        return boxes, scores, classes, num   

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        boxes, scores, classes, num = self.inference(image)
        max_idx = np.argmax(scores[0])
        max_score = np.max(scores[0])
        class_pred_single = classes[0][max_idx]
        candidate_classes = classes[0][scores[0]>0.1]

        if len(candidate_classes)==0:
                return TrafficLight.UNKNOWN

        nb_red = sum(candidate_classes == 1.)
        nb_green = sum(candidate_classes == 2.)
        nb_yellow = sum(candidate_classes == 3.)
        nb_classes = np.array([nb_red, nb_green, nb_yellow])

        class_pred = np.argmax(nb_classes)
        return CLASS_MAP[class_pred]

        
