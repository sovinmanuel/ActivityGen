import os
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch


class OneShotDetector:
    def __init__(self, model_id="google/owlvit-large-patch14"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.processor = OwlViTProcessor.from_pretrained(model_id)

    def image_guided_detection(
        self, image, query_image, score_threshold=0.7, nms_threshold=0.3
    ):
        target_sizes = torch.Tensor([image.size[::-1]])
        inputs = self.processor(
            query_images=query_image, images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.image_guided_detection(**inputs)

        outputs.logits = outputs.logits.cpu()
        outputs.target_pred_boxes = outputs.target_pred_boxes.cpu()

        results = self.processor.post_process_image_guided_detection(
            outputs=outputs,
            threshold=score_threshold,
            nms_threshold=nms_threshold,
            target_sizes=target_sizes,
        )
        boxes, scores = results[0]["boxes"], results[0]["scores"]

        return boxes, scores

    def detect_objects_folder(
        self, base_image_path, query_folder_path, score_threshold=0.7, nms_threshold=0.3
    ):
        base_image = Image.open(base_image_path).convert("RGB")
        query_images = [
            Image.open(os.path.join(query_folder_path, filename)).convert("RGB")
            for filename in os.listdir(query_folder_path)
        ]

        detections = []
        for query_image, query_name in zip(query_images, os.listdir(query_folder_path)):
            boxes, scores = self.image_guided_detection(
                base_image, query_image, score_threshold, nms_threshold
            )
            detection = {"query_name": query_name, "detections": []}
            for i, (box, score) in enumerate(zip(boxes, scores)):
                detection["detections"].append(
                    {
                        "id": i + 1,
                        "height": box[3] - box[1],
                        "width": box[2] - box[0],
                        "position": {
                            "x1": box[0],
                            "y1": box[1],
                            "x2": box[2],
                            "y2": box[3],
                        },
                        "score": score.item(),
                    }
                )
            detections.append(detection)

        return detections
