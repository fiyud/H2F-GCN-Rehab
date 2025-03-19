DATA_PATH = 'Kimore'
EXERCISE = 5  # Default to exercise 5
CHUNK_SIZE = 50  # Size of sequence chunks

MODEL = 'three_stream'
HIDDEN_DIM = 128
NUM_LAYERS = 3
NUM_HEADS = 4
DROPOUT = 0.15

EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
SEED = 100

VISUALIZE = True
VIS_RATIO = 0.3

EDGE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), 
    (4, 5), (5, 6), (6, 7), 
    (8, 9), (9, 10), (10, 11), 
    (0, 12), (12, 13), (13, 14), (14, 15), 
    (0, 16), (16, 17), (17, 18), (18, 19), 
    (2, 20), (7, 21), (6, 22), 
    (11, 23), (10, 24)
]

KINECT_JOINTS = [
    "spinebase", "spinemid", "neck", "head", 
    "shoulderleft", "elbowleft", "wristleft", 
    "handleft", "shoulderright", "elbowright", 
    "wristright", "handright", "hipleft", "kneeleft", 
    "ankleleft", "footleft", "hipright", "kneeright", 
    "ankleright", "footright", "spineshoulder", "handtipleft", 
    "thumbleft", "handtipright", "thumbright"
]