# Annotations
YOLO format annotation files for training/validation.

Each annotation file should have the same name as the image/frame and .txt extension.

Format per line: <class_id> <x_center> <y_center> <width> <height>
All values normalized to 0-1 range.

Classes:
0 - forklift
1 - pallet
2 - person
