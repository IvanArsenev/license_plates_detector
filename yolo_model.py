"""License plate detection and training using YOLO model."""

import os
import argparse
from collections import defaultdict
import logging
import cv2
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class YoloLicensePlateDetector:
    """YOLO model for license plate detection and training."""

    def __init__(self):
        """Initialize YOLO model with tracking and counting parameters."""
        self.tracked_plates = defaultdict(lambda: {'last_seen': 0, 'counted': False})
        self.plate_counter = 0
        self.frame_counter = 0
        self.max_frames_to_forget = 30

    def process_frame(self, frame, model, license_plate_class):
        """Process a single frame for license plate detection."""
        results = model.track(frame, persist=True)
        annotated_frame = frame.copy()
        current_frame_plates = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, class_id in zip(boxes, track_ids, classes):
                if class_id == license_plate_class:
                    self._process_license_plate(
                        box, track_id, annotated_frame, current_frame_plates
                    )

        return annotated_frame

    def _process_license_plate(self, box, track_id, frame, current_frame_plates):
        """Process a detected license plate."""
        x1, y1, x2, y2 = map(int, box)
        current_frame_plates.add(track_id)

        if (track_id not in self.tracked_plates or
                (self.frame_counter - self.tracked_plates[track_id]['last_seen'])
                > self.max_frames_to_forget):
            if not self.tracked_plates[track_id]['counted']:
                self.plate_counter += 1
                self.tracked_plates[track_id]['counted'] = True
                logging.info(
                    "New license plate detected! Total count: %d",
                    self.plate_counter
                )

        self.tracked_plates[track_id]['last_seen'] = self.frame_counter

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    def run_demo(self, model_path, license_plate_class, video_path):
        """
        Run license plate detection demo on a video.

        Args:
            model_path (str): Path to the YOLO model.
            license_plate_class (int): Class ID for license plates.
            video_path (str): Path to the video file.
        """
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logging.error("Error when opening the video.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info("Video processing completed.")
                break

            self.frame_counter += 1
            annotated_frame = self.process_frame(frame, model, license_plate_class)

            if self.frame_counter % 30 == 0:
                logging.info(
                    "Current frame: %d, Unique plates detected: %d",
                    self.frame_counter,
                    self.plate_counter
                )

            cv2.imshow('YOLO Live Video - License Plates', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Final count of unique license plates: %d", self.plate_counter)

    def train_model(self, training_config):
        """
        Train the YOLO model.

        Args:
            training_config (dict): Configuration dictionary containing:
                - data_path (str): Path to the data configuration file
                - model_path (str): Path to the base model
                - results_path (str): Path to save training results
                - epochs (int): Number of training epochs
                - batch_size (int): Batch size for training
                - img_size (int): Image size for training
        """
        os.makedirs(training_config['results_path'], exist_ok=True)
        model = YOLO(training_config['model_path'])
        model.train(
            data=training_config['data_path'],
            epochs=training_config['epochs'],
            batch=training_config['batch_size'],
            imgsz=training_config['img_size'],
            project=training_config['results_path'],
            name='yolo_training',
            exist_ok=True
        )
        logging.info("Training completed. The model is saved in the results directory")


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="YOLO Pipeline CLI")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["demo", "train"],
        help="Select the mode: demo or train"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model"
    )
    parser.add_argument(
        "--license_plate_class",
        type=int,
        default=0,
        help="License plate class ID"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        help="Path to the video file"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the data.yaml file for training"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="results",
        help="Path to save training results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=72,
        help="Image size for training"
    )

    args = parser.parse_args()
    detector = YoloLicensePlateDetector()

    if args.mode == "demo":
        if args.video_path is None:
            logging.error("Video path must be specified in demo mode")
        else:
            detector.run_demo(args.model_path, args.license_plate_class, args.video_path)

    elif args.mode == "train":
        if args.config_path is None:
            logging.error("Data path must be specified in train mode")
        else:
            training_config = {
                'data_path': args.config_path,
                'model_path': args.model_path,
                'results_path': args.results_path,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'img_size': args.img_size
            }
            detector.train_model(training_config)


if __name__ == "__main__":
    main()
