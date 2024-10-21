
# FuturesNet: Capturing Patterns of Price Fluctuations in Domestic Futures Trading

This repository contains the official implementation of the paper **"FuturesNet: Capturing Patterns of Price Fluctuations in Domestic Futures Trading."** FuturesNet is a deep learning architecture designed to capture both short-term and long-term patterns in futures trading data. The model leverages a combination of Inception modules, Long Short-Term Memory (LSTM) networks, and autoregressive components to enhance the prediction accuracy of futures price fluctuations.

## Project Structure

- **`main.py`**: The main entry point to run the model.
- **`train.py`**: Script for training the FuturesNet model.
- **`process_data.py`**: Handles data preprocessing steps.
- **`models/`**: Contains model definitions.
- **`utils.py`**: Utility functions used across the project.
- **`collect.py`**: Data collection script.
- **`visualize/`**: Scripts for visualizing model performance and outputs.
- **`analyze/`**: Contains analysis scripts to evaluate model predictions.

## Requirements

The code requires the following libraries:

- `Python 3.7+`
- `TensorFlow` or `PyTorch` (depending on the model definition)
- `NumPy`
- `Pandas`
- `Matplotlib`

You can install the dependencies by running:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the FuturesNet model on your dataset, use the following command:

```bash
python train.py --data_dir ./newdata --save_dir ./save
```

### Running the Model
To run the model and predict price fluctuations using preprocessed data, execute the following command:

```bash
python main.py --data_dir ./newdata --model_dir ./models/futuresnet_model
```

### Data Preprocessing
If you need to preprocess your data before training or testing, you can use the following command:

```bash
python process_data.py --input_dir ./rawdata --output_dir ./newdata
```

### Visualizing Results
Once the model has been trained and evaluated, you can visualize the predictions and other statistics using the scripts in the `visualize/` directory.

```bash
python visualize/plot_results.py --results_dir ./output
```

## Dataset

The dataset used in this project consists of domestic futures trading data. Please refer to the paper for detailed descriptions of the dataset and features used. For custom datasets, ensure that the data is preprocessed in a format similar to the original dataset structure.

## Citation

If you find this code useful for your research, please consider citing our paper:

```
@article{FuturesNet2024,
  title={FuturesNet: Capturing Patterns of Price Fluctuations in Domestic Futures Trading},
  author={Author Names},
  journal={Journal Name},
  year={2024},
}
```
