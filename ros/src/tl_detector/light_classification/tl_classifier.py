from styx_msgs.msg import TrafficLight
# from keras.models import load_model,Model, model_from_json
# from keras.applications.mobilenet import MobileNet, relu6, DepthwiseConv2D
# from keras.layers import Dense, GlobalAveragePooling2D
# from keras.utils.generic_utils import CustomObjectScope
import numpy as np
import os
import cv2
import tensorflow as tf
import rospy
# from object_detection.utils import label_map_util

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        # Set pathes
        basepath = os.path.dirname(os.path.abspath(__file__))
        path_to_model = os.path.join(basepath, 'ssd_mobilenet_300')
        # path_to_model = os.path.join(basepath, 'ssd_mobilenet_newdata')
        # path_to_model = os.path.join(basepath, 'ssd_inception_v2_512')
        # path_to_model = os.path.join(basepath, 'faster_rcnn_inception')
        # path_to_model = os.path.join(basepath, 'frozen_graph_v6')
        path_to_graph = os.path.join(path_to_model, 'frozen_inference_graph.pb')
        path_to_labels = os.path.join(path_to_model, 'label_map.pbtxt')
        
        self.detection_graph = self.load_graph(path_to_graph)
        self.tf_config = tf.ConfigProto()
        # self.tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.detection_graph, config=self.tf_config)
        # Run fake data during init to warm up TensorFlow's memory allocator
        warmup_iter = 5
        for iter in range(warmup_iter):
            synth_data = np.random.randint(low=0, high=255, size=(600, 800, 3), dtype=np.uint8)
            self.inference(synth_data)
        rospy.loginfo("Traffic Light detector bootstrap executed")


        # NUM_CLASSES =4
        # label_map = label_map_util.load_labelmap(path_to_labels)
        # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        # self.category_index = label_map_util.create_category_index(categories)
            
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
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
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
        # resized_img = np.expand_dims(cv2.resize(image, (128,128)), axis=0)
        # resized_img  = self.standardization(resized_img)
        # with self.graph.as_default():
        #      y_pred = self.model.predict(resized_img)

        # if np.max(y_pred)>0.7:
        #     return np.argmax(y_pred)

        # with self.detection_graph.as_default():
        #     with tf.Session(graph=self.detection_graph) as sess:
                # image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # detect_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # detect_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                # detect_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                # num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                
                # image_np = self.load_image_into_numpy_array(image)
                # image_expanded = np.expand_dims(image, axis=0)
                    
                # (boxes, scores, classes, num) = sess.run(
                #     [detect_boxes, detect_scores, detect_classes, num_detections],
                #     feed_dict={image_tensor: image_expanded})
        boxes, scores, classes, num = self.inference(image)
        max_idx = np.argmax(scores[0])
        max_score = np.max(scores[0])
        class_pred_single = classes[0][max_idx]
        # rospy.loginfo("predicted classes", classes[0])
        # print(classes[0])
        # print(scores[0])
        candidate_classes = classes[0][scores[0]>0.1]

        if len(candidate_classes)==0:
            # if max_score>=0.01:
            #     candidate_classes = classes[0][scores[0]>=0.01]
            # else:
                return TrafficLight.UNKNOWN

        # print(class_candidate)
        nb_red = sum(candidate_classes == 1.)
        nb_green = sum(candidate_classes == 2.)
        nb_yellow = sum(candidate_classes == 3.)
        nb_classes = np.array([nb_red, nb_green, nb_yellow])
        # print(nb_classes)
        class_pred = np.argmax(nb_classes)
        # if scores[0][max_idx]>=0.02:
            # categories[max_idx]['name']
        if class_pred == 0:
            return TrafficLight.RED
        elif class_pred == 1:
            return TrafficLight.GREEN
        elif class_pred == 2:
            return TrafficLight.YELLOW
        #TODO implement light color prediction
        
