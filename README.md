# Disjoint-CNN for Multivariate Time Series Classification

#### Authors: [Navid Mohammadi Foumani](https://www.linkedin.com/in/navid-foumani/), [Chang Wei Tan](https://changweitan.com/), [Mahsa Salehi](https://research.monash.edu/en/persons/mahsa-salehi)
#### ConvTran Paper: [ICDM](https://ieeexplore.ieee.org/document/9679860)
This is a Tensorflow implementation of Disjoint-CNN for Multivariate Time Series Classification.
Additionally, a PyTorch implementation of the **1+1D** block is also included.

## Datasets
We evaluated the Disjoint-CNN model using of 30 datasets from the UEA archive.
### Manual download:

You should manually download the datasets using the provided link and place them into the pre-made directory.
UEA: http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip
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


