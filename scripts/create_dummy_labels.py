import os

splits = ['train', 'test']
base_dir = 'C:/Users/Student/Desktop/Coding_results'

for split in splits:
    split_dir = os.path.join(base_dir, split)
    rgb_dir = os.path.join(split_dir, 'rgb')
    labels_dir = os.path.join(split_dir, 'labels')
    os.makedirs(labels_dir, exist_ok=True)
    
    if os.path.exists(rgb_dir):
        for f in os.listdir(rgb_dir):
            if f.endswith('.png'):
                stem = f.replace('.png', '.txt')
                with open(os.path.join(labels_dir, stem), 'w') as out_f:
                    out_f.write("0 0.5 0.5 0.2 0.2\n") # Dummy bounding box: class_id, x_center, y_center, width, height
print("Dummy labels created.")
