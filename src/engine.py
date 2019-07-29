import csv
import os
import numpy as np
import tensorflow as tf
import pickle
from src.utils import util
from src.utils.util import log
from src.builders import model_builder, data_builder



class BaseEngine(object):

    def __init__(self, config_name, tag):
        self.tag = util.generate_tag(tag)

        # assign configuration
        config = util.load_config(config_name)
        self.model_config = config['model']
        self.eval_config = config['eval']
        self.data_config = config['data']

        # misc information
        self.model_name = self.model_config['name']

        # setup a directory to store evaluation results
        util.setup(self.model_name, self.tag)

    def predict(self):
        raise NotImplementedError

    def evaluation(self):
        raise NotImplementedError


class Engine(BaseEngine):

    def __init__(self, config_name, tag):
        super(Engine, self).__init__(config_name, tag)

        # build dataloader
        self.dataloader = data_builder.build(self.data_config)

        # build model
        self.model = model_builder.build(self.model_config)

    def predict(self):
        threshold = self.eval_config['threshold']
        label_type = self.data_config['label_type']
        detection_graph = self.model.detection_graph

        image_paths, boxes, scores, classes, labels = [], [], [], [], []
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
                score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
                class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

                for image_path_batch, label_batch in self.dataloader:
                    image_batch = util.load_images(image_path_batch)

                    (box_batch, score_batch, class_batch) = sess.run(
                        [box_tensor, score_tensor, class_tensor],
                        feed_dict={image_tensor: image_batch})

                    if self.eval_config['store_detection_results']:
                        util.store_detections(
                            image_batch, box_batch, score_batch, class_batch, threshold)

                    image_paths.append(image_path_batch)
                    boxes.append(box_batch)
                    scores.append(score_batch)
                    classes.append(class_batch)
                    labels.append(label_batch)

        image_paths = np.concatenate(image_paths, axis=0)
        boxes = np.concatenate(boxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        classes = np.concatenate(classes, axis=0)
        labels = np.concatenate(labels, axis=0)

        #information = {'paths': image_paths, 'classes': classes,
        #               'scores': scores, 'boxes': boxes}
        #information_path = 'information_test.pkl'
        #with open(information_path, 'wb') as f:
        #    pickle.dump(information, f)

        num_images = len(image_paths)
        predictions = self._classify(
            scores, classes, num_images, threshold, label_type)

        return image_paths, predictions, labels, label_type


    def _classify(self, scores, classes, num_images, threshold, label_type):
        predictions = []

        for i in range(0, num_images):
            above_threshold = scores[i] > threshold
            classes_above_threshold = np.extract(classes[i], above_threshold)
            scores_above_threshold = np.extract(scores[i], above_threshold)

            if label_type == 'binary':
                is_animal = np.sum(classes_above_threshold == 1) > 0
                predictions.append(int(is_animal))
            else:
                dominant_class_dict = {}
                for cls, score in zip(classes_above_threshold,
                                        scores_above_threshold):
                    if cls not in  dominant_class_dict:
                        dominant_class_dict[cls] = score
                    else:
                        dominant_class_dict[cls] += score

                dominant_class = max(dominant_class_dict, key=dominant_class_dict.get)
                predictions.append(int(dominant_class))

        return predictions


    def evaluate(self):
        image_paths, predictions, labels, label_type = self.predict()
        is_label_available = util.is_label_available(labels)

        if is_label_available:
            num_images = len(labels)
            num_corrects = np.sum(predictions == labels)
            accuracy = num_corrects / float(num_images)
            log.infov('Evaluation accuracy: {}'.format(accuracy))
        else:
            ids = [path.split('/')[-1].split('.')[0] for path in image_paths]
            results = np.stack((ids, predictions), axis=-1)
            results = np.concatenate(([['id', 'animal_present']], results), axis=0)
            output_file = os.path.join(self.eval_config['output_dir'], 'results.csv')
            with open(output_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(results)
            log.infov('Evaluation results are written on {}'.format(output_file))
        return None

