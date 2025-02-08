# Experiment: Boolean Function Training

## Objective

Learn how transformers with non-monotonic targets learn over two head dimensions, and ensure that we can interpret that behavior.

To do this, we define a language called Double Majority which takes two thresholds, $\alpha_1$ and $\alpha_2$, and contains strings of length $n$ for which the number of 1s is in the interval $[\alpha_1 n,\alpha_2 n]$.

## Transformer Model

Our transformer is a single layer, single attention head transformer with no MLP layer, no positional encoding, and no residual stream. Thus, it consists of the following parameters: 

* Attention Head
    * $W_Q$: shape (head_dim, hidden_dim)
    * $W_K$: shape (head_dim, hidden_dim)
    * $W_V$: shape (head_dim, hidden_dim)
    * $W_O$: shape (hidden_dim, head_dim)

Given a bitstring $x$, we encode it as a (func_width, 3) sized matrix by one-hot encoding the binary vocabulary (with an extra token for the [CLS] token embedded at the beginning). As an example, the string 001101 would be encoded as the matrix

$$X = \begin{pmatrix}
0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 
0 & 0 & 0 & 1 & 1 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}^T$$

Let $h$ be the head dimension (denoted head_dim below). After encoding, the attention head performs the following calculation to get the outut sequence 

$$
A = \text{softmax}\left(\dfrac{(XW_Q^T)(XQ_K^T)^T}{\sqrt{h}}\right)
$$

$$
Y = AXW_O^T
$$

$Y$ has shape (func_width, 3), so we just consider the output corresponding to the cls token (the first row), giving a 3 dimensional vector. The first two components of that vector are taken as the logits for the 0 class and 1 class, respectively. 

## Experiment Setup

To create a training and validation dataset for the transformer, we sampled bitstrings of a fixed length and annotated them with a binary classification target encoding whether more than half of the sampled bits in a string were 1. 

See the hyperparameters use for our experiment here:

| Hyperparamter | Value | Explanation |
| ------------- | ----- | ----------- |
| `lr`          | 0.01  | Learning Rate |
| `dataset_size`| 2000  | Number of bitstrings to sample |
| `func_width` | 10 | Length of bitstrings (functions) |
| `head_dim` | 2 | Hidden dim of each attention head of the transformer | 
| `num_epochs` | 1000 | Number of epochs | 
| `train_frac` | 0.4 | Fraction of sampled dataset to be used for training (the rest will be used for testing) |
| `low_threshold` | 0.4 | Fraction of bits which must be 1 for label to change from 0 to 1
| `high_threshold` | 0.7 | Fraction of bits which must be 1 for label to change from 1 back to 0 |

Note that we used a learning rate scheduler with the following configuration: 

```ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)```


## Training Results

As described above, we trained the model for `num_epochs` epochs 

![Training Results](train_results_fig.svg)

As shown in the figure, the model fails to learn double majority.

## Interpretation

![Parameter View](parameter_view_fig.svg)

Note only some of these numbers are consequential (i.e. only some are in the computational path of the network):

1. Only the query at index 2 matters because we are only reading from the output of the [CLS] token (token id 2).
2. As in any transformer, we can combine the w_k and w_v matrices because they always operate together. In this case that just comes down to elementwise multiplication because they are just vectors. 

Thus, the Effective QKV matrix above represents the linear transformation from the vector $[n_0, n_1, n_{CLS}]$ to the 2-dimensional hidden vector of the attention head. The $W_O$ weight above represents the linear transformation from the 2-dimensional hidden vector to the logit vector $[y_0, y_1, y_{CLS}]$. Effective QKVO represents the composition of these two transformations. The output of the network is controlled by $\text{sgn}(y_1-y_0)$ because we simply compare which is greater to select the output. Because $\text{sgn}$ and all linear transformations are monotonic, this model is monotonic in $[n_0, n_1, n_{CLS}]$, but the function we are trying to model isn't. Thus, it is impossible for a transformer of this type to learn DOUBLE MAJORITY.

The above interpretation suppressed the Softmax function, but it is also monotonic so it's insertion does not affect the overall model's monotonicity.