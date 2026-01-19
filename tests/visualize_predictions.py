import argparse
import os
import cv2
import glob
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Visualize predictions from txt files")
    parser.add_argument('--txt-dir', type=str, required=True, help='Directory with prediction txt files')
    parser.add_argument('--img-dir', type=str, default=r"datasets/D-Fire/test/images", help='Directory with original images') 
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    # Auto-determine output if not specified
    if args.output_dir is None:
        # If txt-dir is Results_test/{model}/images/labels, result should be Results_test/{model}/images/visualized
        if "labels" in args.txt_dir:
            args.output_dir = args.txt_dir.replace("labels", "visualized")
        else:
             args.output_dir = "visualized_predictions"

    os.makedirs(args.output_dir, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(args.txt_dir, '*.txt'))
    print(f"Found {len(txt_files)} prediction files.")
    
    if not txt_files:
        print(f"No txt files found in {args.txt_dir}")
        return

    for txt_path in tqdm(txt_files):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        # Parse image name form first line: "Image: filename.jpg"
        img_line = lines[0].strip()
        if not img_line.startswith("Image: "):
            # Try to infer from filename if header missing (fallback)
            base = os.path.basename(txt_path).replace('.txt', '')
            # Try extensions
            for ext in ['.jpg', '.png', '.jpeg']:
                if os.path.exists(os.path.join(args.img_dir, base + ext)):
                    img_name = base + ext
                    break
            else:
                 print(f"Skipping {txt_path}, invalid header and cannot matching image.")
                 continue
        else:    
            img_name = img_line.replace("Image: ", "")
            
        img_path = os.path.join(args.img_dir, img_name)
        
        # Check if img_path exists, maybe try variations if img_dir + img_name fails
        if not os.path.exists(img_path):
             # Try searching in img-dir recursively or check naming
             print(f"Image {img_path} not found.")
             continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Parse boxes
        # Lines 1+: cls x1 y1 x2 y2 conf
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            cls_name = parts[0]
            try:
                x1, y1, x2, y2, conf = map(float, parts[1:6])
            except ValueError:
                continue
            
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            
            # Draw
            color = (0, 0, 255) if 'fire' in cls_name.lower() else (255, 0, 0) # Red for fire, Blue for smoke
            cv2.rectangle(img, p1, p2, color, 2)
            label = f"{cls_name} {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            c2 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
            cv2.rectangle(img, p1, c2, color, -1)
            cv2.putText(img, label, (p1[0], p1[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        out_path = os.path.join(args.output_dir, img_name)
        cv2.imwrite(out_path, img)
        
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main()
