# H2F-GCN Rehab

bash```
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
│       ├── hyper_compute.py       
│       ├── tdgc.py               
│       └── temporal_conv.py        
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