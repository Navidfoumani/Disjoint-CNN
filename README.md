# Disjoint-CNN for Multivariate Time Series Classification

#### Authors: [Navid Mohammadi Foumani](https://www.linkedin.com/in/navid-foumani/), [Chang Wei Tan](https://changweitan.com/), [Mahsa Salehi](https://research.monash.edu/en/persons/mahsa-salehi)
#### ConvTran Paper: [ICDM21](https://ieeexplore.ieee.org/document/9679860)
This is a Tensorflow implementation of Disjoint-CNN for Multivariate Time Series Classification.
Additionally, a PyTorch implementation of the **1+1D** block is also included.

## Overview 
<p align="justify">
  
Existing models consider a time series as a 1-Dimensional (1D) image and employ 1D convolution operations to extract features from the multivariate time series. However, these models do not consider the importance of the interaction between channels. In this work, we challenge this view and introduce a convolution block called **1+1D** that emphasizes the interaction between input channels. The **1+1D** block explicitly factorizes 1D convolution into two unmixed and successive operations: 1D temporal convolution per channel and 1D spatial convolution that learns the interaction between the channels through the features extracted from the temporal convolution.
</p>

### 1D convolution:

<pre>
<code>
# Input shape: (Series_length, Channel)
conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(input)
conv1 = BatchNormalization()(conv1)
conv1 = Activation(activation='relu')(conv1)
</code>
</pre>
### 1+1D convolution:
<pre>
<code>
# Input shape: (Series_length, Channel, 1)
# Temporal Convolutions
conv1 = Conv2D(64, (8, 1), strides=1, padding="same", kernel_initializer='he_uniform')(input)
conv1 = BatchNormalization()(conv1)
conv1 = ELU(alpha=1.0)(conv1)

# Spatial Convolutions
conv1 = Conv2D(64, (1, input_shape[1]), strides=1, padding="valid", kernel_initializer='he_uniform')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = ELU(alpha=1.0)(conv1)
conv1 = Permute((1, 3, 2))(conv1)
</code>
</pre>

Please note that in the 1+1D convolution implementation, `input` represents the input tensor for the 1+1D block, and `input_shape` refers to the shape of the input series.
There are two main benefits to this decomposition. 
<p align="justify">
First, we can use an additional nonlinear activation function between these two operations, providing the model with an additional nonlinear representation. This doubling of nonlinear functions allows the model to extract more complex functions with fewer parameters, while keeping the number of parameters approximately the same as typical 1D convolution operations.
</p>

<p align="justify">
Second, this decomposition facilitates the optimization process in our deep neural network model. Compared to typical 1D filters where temporal and spatial filters are intertwined, the 1+1D blocks (with decomposed temporal and spatial components) are easier to optimize, resulting in lower test and training loss in practice.
</p>


In this repository, we reimplement the following algorithms for comparison with Disjoint CNN:
- **Disjoint CNN (DCNN_2L, DCNN_3L, DCNN_4L):** These models implement Disjoint CNN and are designed specifically for MTSC.
- **Multi-channel Deep Convolutional Neural Network (MC_CNN):** This pioneer model uses CNN for MTSC.
- **Fully Convolutional Network (FCN):** FCN is one of the most accurate deep neural networks for MTSC.
- **Residual Network (ResNet):** ResNet is a highly accurate deep neural network suitable for both univariate TSC and MTSC.
- **Multivariate LSTM-FCN (MLSTM_FCN):** This model combines LSTM and CNN to capture sequential information in time series.

- **Temporal CNN (T_CNN):** This model focuses on temporal convolutions for feature extraction.
- **Spatial CNN (S_CNN):** This model emphasizes spatial convolutions for feature extraction.
- **Spatial-Temporal CNN (ST_CNN):** This model combines both spatial and temporal convolutions for feature extraction.
- **Disjoint FCN (D_FCN):** This model is an extension of FCN with Disjoint architecture for MTSC.
- **Disjoint ResNet (D_ResNet):** This model is an extension of ResNet with Disjoint architecture for MTSC.





## Datasets
We evaluated the Disjoint-CNN model using of 30 datasets from the UEA archive.
### Manual download:

You should manually download the datasets using the provided link and place them into the pre-made directory.

**UEA**: http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip

Copy the datasets folder to: Multivariate_ts/<Dataset Name>

  
## Installation

_Instructions refer to Unix-based systems (e.g. Linux, MacOS)._

This code has been tested with `Python 3.7` and `3.8`.

`pip install -r requirements.txt`

## Usage

In `Main.py` you can select the datasets and modify the model parameters.


## Citation

If you find *ConvTran* useful for your research, please consider citing this paper:

````
```
@misc{foumani2023improving,
      title={Improving Position Encoding of Transformers for Multivariate Time Series Classification}, 
      author={Navid Mohammadi Foumani and Chang Wei Tan and Geoffrey I. Webb and Mahsa Salehi},
      year={2023},
      eprint={2305.16642},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
````


