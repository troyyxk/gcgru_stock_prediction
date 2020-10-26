# Forecasting Stock Prices Using Stock Correlation Graph
We create a new modle by combining GCN model with GRU model and achieve around 5% incraese in accuracy in comparison to baseline model.

## Model Architecture
![GCGRU Structure](image/model.jpeg)

## Illustration of Graph Convolution
![Graph Convolution](image/gcc.jpeg)

## Dependencies
  * Tensorflow
  * Pandas
  * Numpy
  * Sklearn
  * Configparser
  
## Workflow
The workflow is utralize. Start training the model by simply run:
```
python3 train.py
```

Hyper parameters can be changed at the file 

```
config.ini
```
### Hyper Parameter

## Directory Structure

```
gcgru_stock_prediction
├── config.ini
├── gcgru.py
├── input_data.py
├── train.py
├── utils.py
├── image
│   ├── gcc.jpeg
│   └── model.jpeg
└── data
    ├── adj
    │   ├── dow
    │   │   ├── 1day
    │   │   │   ├── dow_1day_050_01_corr.csv
    │   │   │   ├── dow_1day_055_01_corr.csv
    │   │   │   └── ...
    │   │   └── ...
    │   └── etf
    │       ├── 1day
    │       │   ├── etf_1day_050_01_corr.csv
    │       │   ├── etf_1day_055_01_corr.csv
    │       │   └── ...
    │       └── ...
    └── data
        ├── dow
        |   ├── dow_10min_price.csv
        |   ├── dow_15min_price.csv
        |   ├── dow_1day_price.csv
        |   ├── dow_1h_price.csv
        |   └── dow_30min_price.csv
        └── etf
            ├── etf_15min_price.csv
            ├── etf_1day_price.csv
            ├── etf_1h_price.csv
            └── etf_30min_price.csv
```
