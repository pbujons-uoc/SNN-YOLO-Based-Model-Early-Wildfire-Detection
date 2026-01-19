
import os
import shutil
import yaml

# Paths
ROOT_DIR = os.getcwd()
DATASET_DIR = os.path.join(ROOT_DIR, 'datasets', 'dummy_gen1')
IMAGES_TRAIN = os.path.join(DATASET_DIR, 'images', 'train')
IMAGES_VAL = os.path.join(DATASET_DIR, 'images', 'val')
LABELS_TRAIN = os.path.join(DATASET_DIR, 'labels', 'train')
LABELS_VAL = os.path.join(DATASET_DIR, 'labels', 'val')

SOURCE_PT = os.path.join(ROOT_DIR, 'AoF00000.pt')
SOURCE_LABEL = os.path.join(ROOT_DIR, 'datasets', 'D-Fire', 'train', 'labels', 'AoF00000.txt')

def main():
    if not os.path.exists(SOURCE_PT):
        print(f"Error: {SOURCE_PT} not found.")
        return
    if not os.path.exists(SOURCE_LABEL):
        print(f"Error: {SOURCE_LABEL} not found.")
        return

    # Create directories
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        os.makedirs(d, exist_ok=True)
        print(f"Created {d}")

    # Copy files
    # Copy pt to images/train and images/val
    shutil.copy(SOURCE_PT, os.path.join(IMAGES_TRAIN, 'AoF00000.pt'))
    shutil.copy(SOURCE_PT, os.path.join(IMAGES_VAL, 'AoF00000.pt'))
    print("Copied AoF00000.pt to images/train and images/val")

    # Copy txt to labels/train and labels/val
    shutil.copy(SOURCE_LABEL, os.path.join(LABELS_TRAIN, 'AoF00000.txt'))
    shutil.copy(SOURCE_LABEL, os.path.join(LABELS_VAL, 'AoF00000.txt'))
    print("Copied AoF00000.txt to labels/train and labels/val")

    # Create data.yaml
    data_yaml = {
        'path': DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'nc': 2,
        'names': ['smoke', 'fire'] # 0: smoke, 1: fire based on D-Fire data.yaml
    }
    
    yaml_path = os.path.join(DATASET_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print(f"Created {yaml_path}")

if __name__ == '__main__':
    main()
