import struct
import numpy as np

def parse_idx1_file(file_path):
    with open(file_path, 'rb') as f:
        # Read header
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}. Expected 2049 for IDX1 files.")
        
        # Read label data
        label_data = np.frombuffer(f.read(), dtype=np.uint8)
    
    return label_data

labels = parse_idx1_file("C:\\Users\\ashwa\\projects\\cnn_from_scratch\\archive (1)\\train-labels.idx1-ubyte")
