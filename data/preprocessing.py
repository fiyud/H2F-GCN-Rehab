import numpy as np
import torch

def preprocess_data_and_labels(df, chunk_size):
    all_sequences = []
    all_labels = []

    for _, row in df.iterrows():
        num_frames = row.iloc[0].shape[0]  
        num_chunks = num_frames // chunk_size  

        labels = row[['cTS']].values.astype(np.float32)  

        for i in range(num_chunks):
            segment = np.array([joint[i * chunk_size:(i + 1) * chunk_size] for joint in row[:-1]])  
            segment = np.transpose(segment, (1, 0, 2))   # (time_step, num_joints, num_features)
            
            all_sequences.append(segment)
            all_labels.append(labels) 

    data_tensor = torch.tensor(np.array(all_sequences), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(all_labels), dtype=torch.float32)

    return data_tensor, labels_tensor

def get_JCD(p):
    batch_size, frame_num, num_joints, _ = p.shape
    num_pairs = (num_joints * (num_joints - 1)) // 2 # Number of joint pairs
    JCD = torch.zeros((batch_size, frame_num, num_pairs))
    for b in range(batch_size):
        for f in range(frame_num):
            dist_matrix = torch.cdist(p[b, f], p[b, f], p=2)  # p=2 is Euclidean distance
            
            # Extract upper triangular part (excluding main diagonal)
            iu = torch.triu_indices(num_joints, num_joints, offset=1)
            JCD[b, f] = dist_matrix[iu[0], iu[1]]

    JCD = (JCD - JCD.min()) / (JCD.max() - JCD.min())
    return JCD