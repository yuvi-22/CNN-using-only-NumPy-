import struct
import numpy as np
import matplotlib.pyplot as plt

def parse_idx3_file(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}. Expected 2051 for IDX3 files.")
        
        # Read image data
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)  # Reshape to (num_images, rows, cols)
    
    return images


# Parse images
images = parse_idx3_file("C:\\Users\\ashwa\\projects\\cnn_from_scratch\\archive (1)\\train-images.idx3-ubyte")
