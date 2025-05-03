import cv2
import supervision as sv
from ultralytics import YOLOv10

model = YOLOv10(f'best.pt')
image = cv2.imread(r'D:\github\myrepo\yolov10\yolov10_RNE\valid\images\full_frame_20250426_133313_129_png.rf.2aeb1be898bed1bebbe180fa9415a762.jpg')
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)