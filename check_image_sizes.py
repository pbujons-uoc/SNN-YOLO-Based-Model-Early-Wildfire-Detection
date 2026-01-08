"""
Check image sizes in D-Fire dataset.

Analyzes all images in train, val, and test splits.
"""

import cv2
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def check_dataset_sizes(dataset_path):
    """Check sizes of all images in dataset."""
    dataset = Path(dataset_path)
    
    if not dataset.exists():
        print(f"Error: Dataset not found at {dataset}")
        return
    
    splits = ['train', 'val', 'test']
    all_sizes = []
    split_stats = {}
    
    for split in splits:
        images_dir = dataset / split / 'images'
        
        if not images_dir.exists():
            print(f"Warning: {images_dir} not found, skipping")
            continue
        
        # Find all images
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
        
        print(f"\n{'='*60}")
        print(f"Analyzing {split} split ({len(image_files)} images)...")
        print(f"{'='*60}")
        
        sizes = []
        invalid_count = 0
        
        for img_path in tqdm(image_files, desc=f"Processing {split}"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    invalid_count += 1
                    continue
                
                h, w, c = img.shape
                sizes.append((w, h))  # Store as (width, height)
                all_sizes.append((w, h, split))
                
            except Exception as e:
                invalid_count += 1
                print(f"Error reading {img_path}: {e}")
        
        # Statistics for this split
        if sizes:
            size_counter = Counter(sizes)
            unique_sizes = len(size_counter)
            most_common = size_counter.most_common(5)
            
            widths = [w for w, h in sizes]
            heights = [h for w, h in sizes]
            
            split_stats[split] = {
                'total': len(sizes),
                'invalid': invalid_count,
                'unique_sizes': unique_sizes,
                'min_size': (min(widths), min(heights)),
                'max_size': (max(widths), max(heights)),
                'most_common': most_common
            }
            
            print(f"\n{split.upper()} Statistics:")
            print(f"  Total images:     {len(sizes)}")
            print(f"  Invalid images:   {invalid_count}")
            print(f"  Unique sizes:     {unique_sizes}")
            print(f"  Min size (WxH):   {min(widths)}x{min(heights)}")
            print(f"  Max size (WxH):   {max(widths)}x{max(heights)}")
            print(f"\n  Most common sizes:")
            for (w, h), count in most_common:
                percentage = (count / len(sizes)) * 100
                print(f"    {w}x{h}: {count} images ({percentage:.1f}%)")
    
    # Overall statistics
    if all_sizes:
        print(f"\n{'='*60}")
        print("OVERALL DATASET STATISTICS")
        print(f"{'='*60}")
        
        all_dims = [(w, h) for w, h, s in all_sizes]
        size_counter = Counter(all_dims)
        
        widths = [w for w, h, s in all_sizes]
        heights = [h for w, h, s in all_sizes]
        
        print(f"Total images:     {len(all_sizes)}")
        print(f"Unique sizes:     {len(size_counter)}")
        print(f"Min size (WxH):   {min(widths)}x{min(heights)}")
        print(f"Max size (WxH):   {max(widths)}x{max(heights)}")
        
        print(f"\nTop 10 most common sizes:")
        for (w, h), count in size_counter.most_common(10):
            percentage = (count / len(all_sizes)) * 100
            print(f"  {w}x{h}: {count} images ({percentage:.1f}%)")
        
        # Check if all images have the same size
        if len(size_counter) == 1:
            size = list(size_counter.keys())[0]
            print(f"\n All images have uniform size: {size[0]}x{size[1]}")
        else:
            print(f"\n Images have varying sizes ({len(size_counter)} unique dimensions)")
        
        # Save detailed report
        report_file = Path('image_sizes_report.txt')
        with open(report_file, 'w') as f:
            f.write("D-Fire Dataset Image Sizes Report\n")
            f.write("="*60 + "\n\n")
            
            for split in splits:
                if split in split_stats:
                    stats = split_stats[split]
                    f.write(f"{split.upper()} Split:\n")
                    f.write(f"  Total images: {stats['total']}\n")
                    f.write(f"  Invalid images: {stats['invalid']}\n")
                    f.write(f"  Unique sizes: {stats['unique_sizes']}\n")
                    f.write(f"  Min size: {stats['min_size'][0]}x{stats['min_size'][1]}\n")
                    f.write(f"  Max size: {stats['max_size'][0]}x{stats['max_size'][1]}\n")
                    f.write(f"  Most common sizes:\n")
                    for (w, h), count in stats['most_common']:
                        percentage = (count / stats['total']) * 100
                        f.write(f"    {w}x{h}: {count} ({percentage:.1f}%)\n")
                    f.write("\n")
            
            f.write("Overall Statistics:\n")
            f.write(f"  Total images: {len(all_sizes)}\n")
            f.write(f"  Unique sizes: {len(size_counter)}\n")
            f.write(f"  Min size: {min(widths)}x{min(heights)}\n")
            f.write(f"  Max size: {max(widths)}x{max(heights)}\n")
            f.write(f"\n  All unique sizes (sorted by count):\n")
            for (w, h), count in size_counter.most_common():
                percentage = (count / len(all_sizes)) * 100
                f.write(f"    {w}x{h}: {count} ({percentage:.2f}%)\n")
        
        print(f"\nDetailed report saved to: {report_file}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    check_dataset_sizes('datasets/D-Fire')
