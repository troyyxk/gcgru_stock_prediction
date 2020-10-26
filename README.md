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
├── data
│   ├── adj
│   │   ├── dow

│   ├── react-components
│   │   ├── Home
│   │   │   ├── static
│   │   │   │   └── home-books.jpg
│   │   │   ├── index.js
│   │   │   └── styles.css
│   │   ├── Queue
│   │   │   ├── index.js
│   │   │   └── styles.css
│   │   ├── Student
│   │   │   ├── index.js
│   │   │   └── styles.css
│   │   └── ...
│   └── serviceWorker.js
├── package-lock.json
└── src
    ├── actions
    │   └── queue.js
    ├── react-components
    │   ├── Home
    │   │   ├── static
    │   │   │   └── home-books.jpg
    │   │   ├── index.js
    │   │   └── styles.css
    │   ├── Queue
    │   │   ├── index.js
    │   │   └── styles.css
    │   ├── Student
    │   │   ├── index.js
    │   │   └── styles.css
    │   └── ...
    ├── index.js
    ├── index.css
    ├── App.js
    ├── App.css
    └── serviceWorker.js
```
