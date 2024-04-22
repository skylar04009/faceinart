import time
import random
import cv2
import numpy as np
import onnxruntime

from yolov7.utils import xywh2xyxy, nms, draw_detections

cuda = True
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']


class YOLOv7:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, official_nms=False):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.official_nms = official_nms

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=providers)
        # Get model info
        self.get_input_details()
        self.get_output_details()

        self.has_postprocess = 'score' in self.output_names or self.official_nms

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        if self.has_postprocess:
            self.boxes, self.scores, self.class_ids = self.parse_processed_output(outputs)

        else:
            # Process output data
            self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        self.names = ['head', 'person']

        self.colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(self.names)}

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_img = img.copy()
        # Resize input image
        input_img, self.ratio, self.dwdh = self.letterbox(input_img, auto=False)
        input_img = input_img.transpose((2, 0, 1))
        input_img = np.expand_dims(input_img, 0)
        input_img = np.ascontiguousarray(input_img)
        # Scale input pixel values to 0 to 1
        im = input_img.astype(np.float32)
        im /= 255

        return im

    def inference(self, im):
        start = time.perf_counter()
        inp = {self.input_names[0] : im}
        outputs = self.session.run(self.output_names, inp)[0]
        print(outputs)
        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = output
        print(predictions)

        # Get the scores
        scores = predictions[:, 6]

        # Get the class with the highest confidence
        class_ids = predictions[:, 5]

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        print(boxes, class_ids, scores)
        return boxes, scores, class_ids

    def parse_processed_output(self, outputs):

        # Pinto's postprocessing is different from the official nms version
        if self.official_nms:
            scores = outputs[0][:, -1]
            predictions = outputs[0][:, [0, 5, 1, 2, 3, 4]]
        else:
            scores = np.squeeze(outputs[0], axis=1)
            predictions = outputs[1]
        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        if len(scores) == 0:
            return [], [], []

        # Extract the boxes and class ids
        # TODO: Separate based on batch number
        batch_number = predictions[:, 0]
        class_ids = predictions[:, 1].astype(int)
        boxes = predictions[:, 2:]

        # In postprocess, the x,y are the y,x
        if not self.official_nms:
            boxes = boxes[:, [1, 0, 3, 2]]

        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, 1:5]
        result = []
        for i, box in enumerate(boxes):
            box -= np.array(self.dwdh*2)
            box /= self.ratio
            box = box.round().astype(np.int32).tolist()
            result.append(box)
        print(result)
        return result

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, boxes, scores, class_ids):
        ori_images = [image.copy()]
        for box, score, class_id in zip(boxes, scores, class_ids):
            class_id = int(class_id)
            image = ori_images[0]
            name = self.names[class_id]
            color = self.colors[name]
            name += ' ' + str(score)
            cv2.rectangle(image, box[:2], box[2:], color, 2)
            cv2.putText(image, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
        return ori_images[0]


    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    model_path = "best.onnx"

    # Initialize YOLOv7 object detector
    yolov7_detector = YOLOv7(model_path, conf_thres=0.35, iou_thres=0.65)

    img = cv2.imread("../detectAuto/test/WIN_20230511_15_40_52_Pro.jpg")

    # Detect Objects
    boxes, scores, class_ids = yolov7_detector(img)

    # Draw detections
    combined_img = yolov7_detector.draw_detections(img, boxes, scores, class_ids)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
