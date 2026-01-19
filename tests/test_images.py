import argparse
import os
import glob
from tqdm import tqdm
from model_utils import ModelWrapper, project_root, DEFAULT_WEIGHTS


def main():
    parser = argparse.ArgumentParser(description="Test Images with various models")
    parser.add_argument('--model', type=str, nargs='+', required=True, choices=DEFAULT_WEIGHTS.keys(), help='List of models to test')
    parser.add_argument('--weights', type=str, nargs='*', help='Path to weights file. Defaults to known paths. If provided, must match number of models or be single path.')
    parser.add_argument('--source', type=str, default=os.path.join(project_root, 'datasets', 'D-Fire', 'test', 'images'), help='Source directory of images')
    parser.add_argument('--num-images', type=int, default=None, help='Number of images to test. Default: all.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for image selection')
    parser.add_argument('--output-dir', type=str, default="Results_test", help='Base output directory.')
    parser.add_argument('--time-steps', type=int, default=4, help='Time steps for spike encoding (default 4)')
    args = parser.parse_args()
    
    # Handle Weights
    model_weights = {}
    if args.weights:
        if len(args.weights) == 1:
             for m in args.model: model_weights[m] = args.weights[0]
        elif len(args.weights) == len(args.model):
             for i, m in enumerate(args.model): model_weights[m] = args.weights[i]
        else:
             print("Error: Number of weights must match number of models or be 1.")
             return
    else:
        for m in args.model:
            model_weights[m] = DEFAULT_WEIGHTS[m]

    # Get images
    if not os.path.exists(args.source):
        print(f"Error: Source directory {args.source} does not exist.")
        return

    images = glob.glob(os.path.join(args.source, '*'))
    images = [x for x in images if x.lower().endswith(('.jpg', '.png', '.jpeg'))]
    images.sort() # Ensure deterministic order before shuffle

    if args.num_images:
        import random
        random.seed(args.seed)
        random.shuffle(images)
        images = images[:args.num_images]
        print(f"Selected {len(images)} images with seed {args.seed}")
    
    print(f"Total images to process: {len(images)}")
    
    for model_name in args.model:
        print(f"\n--- Processing Model: {model_name} ---")
        weights = model_weights[model_name]
        
        # Determine Output Directory for this model
        # Structure: Results_test/{model}/images/labels
        model_output_dir = os.path.join(args.output_dir, model_name, "images", "labels")
        os.makedirs(model_output_dir, exist_ok=True)

        if not os.path.exists(weights):
            print(f"Warning: Weights file not found at {weights} for {model_name}. Skipping.")
            continue
        
        # Load model
        try:
            wrapper = ModelWrapper(model_name, weights, time_steps=args.time_steps)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue

        class_names = wrapper.get_class_names()
        
        print(f"Running inference on {len(images)} images...")
        
        for img_path in tqdm(images, desc=f"{model_name}"):
            try:
                dets = wrapper.predict(img_path)
            except Exception as e:
                print(f"Error predicting {img_path}: {e}")
                continue
            
            # Save to txt
            base_name = os.path.basename(img_path)
            txt_name = os.path.splitext(base_name)[0] + '.txt'
            txt_path = os.path.join(model_output_dir, txt_name)
            
            with open(txt_path, 'w') as f:
                # First line: Image name
                f.write(f"Image: {base_name}\n")
                for det in dets:
                    # det: [x1, y1, x2, y2, conf, cls]
                    cls_id = int(det[5])
                    # Ensure class name retrieval handles both list and dict
                    if isinstance(class_names, dict):
                         cls_name = class_names.get(cls_id, str(cls_id))
                    elif isinstance(class_names, list):
                         cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
                    else:
                         cls_name = str(cls_id)

                    # Format: class x1 y1 x2 y2 conf
                    line = f"{cls_name} {det[0]:.2f} {det[1]:.2f} {det[2]:.2f} {det[3]:.2f} {det[4]:.2f}\n"
                    f.write(line)
        
        print(f"Predictions for {model_name} saved to {model_output_dir}")

if __name__ == '__main__':
    main()
