# H2F-GCN Rehab

```
kimore/
│
├── requirements.txt                  
├── README.md                    
├── main.py                         
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py              
│   ├── dataset.py                 
│   └── preprocessing.py          
│
├── models/
│   ├── __init__.py
│   ├── gcn.py                       
│   ├── four_stream_gcn.py           
│   ├── three_stream_gcn.py           
│   └── modules/
│       ├── __init__.py
│       ├── graph_conv.py           
│       └── hyper_compute.py       
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                    
│   ├── visualization.py          
│   └── seed.py                    
│
└── configs/
    ├── __init__.py
    └── default_config.py 
```

# Kimore Movement Assessment

This project implements Graph Convolutional Networks (GCNs) for analyzing movement data from the Kimore dataset. The models assess the quality of rehabilitation exercises by predicting clinical assessment scores from motion capture data.

## Features

- Data loading and preprocessing for Kimore dataset
- Multiple GCN-based model architectures:
  - Three-Stream GCN model
  - Four-Stream GCN model
- Hypergraph computation for enhanced feature extraction
- Comprehensive evaluation metrics (MAD, RMSE, MAPE)
- Visualization tools for predictions

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/kimore-gcn.git
cd kimore-gcn
```

2. Install requirements:
```
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run training with default parameters:
```
python main.py --data_path /path/to/Kimore --exercise 5
```

### Model Selection

Choose between different model architectures:
```
python main.py --model three_stream  # Three-Stream GCN (default)
python main.py --model four_stream   # Four-Stream GCN
```

### Training Parameters

Customize training parameters:
```
python main.py --epochs 500 --batch_size 32 --lr 0.001 --hidden_dim 128
```

### Visualization

Enable prediction visualization:
```
python main.py --visualize --vis_ratio 0.5
```

### Save/Load Models

Save and load trained models:
```
python main.py --save_model --model_path my_model.pth
```

## Model Architecture

The project includes two main model architectures:

1. **Three-Stream GCN**: Combines skeleton features, JCD features, and spatial frequency features
2. **Four-Stream GCN**: Adds a temporal frequency stream to the Three-Stream GCN

Each architecture leverages GCN layers, hypergraph computation, and transformer encoders to process motion data effectively.

## Data Processing

The Kimore dataset is processed as follows:
1. Load raw data from CSV and Excel files
2. Extract joint positions and orientations
3. Compute Joint Coordinate Descriptors (JCD)
4. Divide sequences into fixed-size chunks
5. Normalize features and labels

## Results

The models achieve the following metrics on Exercise 5:
- MAD: 0.5191
- RMSE: 0.9774
- MAPE: 1.33%

## License

[Your License Here]

## Acknowledgements

[Your Acknowledgements Here]
