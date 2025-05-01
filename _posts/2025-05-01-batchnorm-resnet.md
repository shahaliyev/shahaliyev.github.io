---
title: Python Code for Batch Normalization and Residual Blocks
description: Based on Andrej Karaphy's makemore and Dive into Deep Learning book chapters 8.5/6. My goal is to introduce breakthroughs which allowed to train deeper neural networks (beyond 100 layers). We explore earlier architectures, such as Inception (GoogleNet) and VGGNet, before going on to discuss Batch Normalization and Residual Blocks. I try to give simple intuition for understanding skip (residual) connection and how it helps with convergence and regularization. I also implement the concepts in PyTorch, while referring to formulae and ideas in the papers.
category: [computer science, deep learning]
---

{% include info.html content="The following material was initially prepared as a lecture for <strong>CSCI 4701: Deep Learning (Spring 2025)</strong> course at ADA University. It is based on Andrej Karpathy's <a href='https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=4'>makemore: part 3</a> and Dive into Deep Learning (<a href='https://d2l.ai'>d2l.ai</a>) book, chapters <a href='https://d2l.ai/chapter_convolutional-modern/batch-norm.html'>8.5</a> and <a href='https://d2l.ai/chapter_convolutional-modern/resnet.html'>8.6</a>. The notebook is the continuation of my lectures on deep learning: <a href='https://shahaliyev.org/writings/backprop'>Python Code from Derivatives to Backpropagation</a>, <a href='https://shahaliyev.org/writings/neural_network'>Python Code from Neuron to Neural Network</a>, <a href='https://shahaliyev.org/writings/cnn-pytorch'> PyTorch Code from Kernel to Convolutional Neural Network</a>, <a href='https://shahaliyev.org/writings/regul-optim'>Python Code for Deep Learning Regularization and Optimization</a>, and <a href='https://shahaliyev.org/writings/nn-ngram'>Python Code for Neural Network N-Gram Model</a>" %}

{% include colab.html link="https://colab.research.google.com/github/shahaliyev/shahaliyev.github.io/blob/main/assets/nb/batchnorm_resnet.ipynb" %}

{% include toc.html show_subheadings=true %}

Increasing the number of layers in neural networks for learning more advanced functions is challenging due to issues like vanishing gradients. [VGGNet](https://arxiv.org/pdf/1409.1556) partially addressed this problem by using repetitive _blocks_ that stack multiple convolutional layers before downsampling with max-pooling. For instance, two consecutive `3x3` convolutional layers achieve the same receptive field as a single `5x5` convolution, while preserving a higher spatial resolution for the next layer. In simpler terms, repeating a smaller kernel allows the network to access the same input pixels while retaining more detail for subsequent processing. Larger kernels blur (downsample) the image more aggressively, which can lead to the loss of important details and force the network to reduce resolution earlier in the architecture and stop.

Despite this breakthrough, VGGNet was still limited and showed diminishing returns beyond `19` layers (hence, `vgg19` architecture). Another architecture was introduced the same year with the paper titled [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842). It was named **Inception** because of the [internet meme](https://knowyourmeme.com/memes/we-need-to-go-deeper) from the infamous _Inception_ movie. I am not joking. If you don't believe me, scroll down the paper for references section and check out the very first reference.

Inception architecture, and its implementation, `GoogLeNet` model (a play on words: 1) was developed by Google researchers, and 2) pays homage to the LeNet architecture), significantly reduced parameter count and leveraged the advantages of the `1x1` convolution kernel (see the [Network in Network](https://arxiv.org/pdf/1312.4400) paper which also introduced `Global Average Pooling (GAP)` layer). Despite enabling deeper networks with far fewer parameters, Inception did not fully resolve the core training and convergence problems faced by very deep models.

[Batch Normalization](https://arxiv.org/pdf/1502.03167) and [Residual Networks](https://arxiv.org/pdf/1512.03385) emerged as two major solutions for efficiently training neural networks as deep as `100` layers and more. We will now set up the data environment and go on discussing the core ideas and implementations of both papers.


```python
import requests
import random
import string
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
```


```python
########## DATA SETUP ##########

url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
response = requests.get(url)
words = response.text.splitlines()
random.shuffle(words)

chars = list(string.ascii_lowercase)
stoi = {ch: i for i, ch in enumerate(chars)}
stoi['<START>'] = len(stoi)
stoi['<END>'] = len(stoi)
itos = {i: ch for ch, i in stoi.items()}

BLOCK_SIZE = 3
VOCAB_SIZE = len(stoi)
EMBED_SIZE = 10
LAYER_SIZE = 100

len(words), BLOCK_SIZE, VOCAB_SIZE, words[0]
```
    # output
    (32033, 3, 28, 'wafi')


A quick sidenote: it is encouraged to split the data into **training**, **validation** (also called _dev_), and **test** sets. When the dataset is not large, an `80/10/10` split is a reasonable ratio for allocation. For larger datasets (e.g. with one million images), it is fine to allocate `90%` or more of your data for training. The training set is used to update the model's _parameters_. The validation set is used for tuning _hyperparameters_ (e.g. testing different learning rates, regularization strengths, etc.). The test split should ideally be used only _once_ to report the final performance of the selected model (e.g. for inclusion in a research paper).


```python
########## DATA PREP ##########

def get_ngrams(start=0, end=None):
  X, Y = [], []
  for word in words[start:end]:
    context = ['<START>'] * BLOCK_SIZE
    for ch in list(word) + ['<END>']:
      X.append([stoi[c] for c in context])
      Y.append(stoi[ch])
      context = context[1:] + [ch]
  return torch.tensor(X), torch.tensor(Y)

def split_data(p=80):
  train_end = int(p/100 * len(words))
  remaining = len(words) - train_end
  val_end = train_end + remaining // 2

  X_train, Y_train = get_ngrams(end=train_end)
  X_val, Y_val = get_ngrams(start=train_end, end=val_end)
  X_test, Y_test = get_ngrams(start=val_end, end=len(words))

  return {
    'train': (X_train, Y_train),
    'val':   (X_val, Y_val),
    'test':  (X_test, Y_test),
  }

data = split_data()

X_train, Y_train = data['train']
X_val, Y_val = data['val']
X_test, Y_test = data['test']

len(X_train), len(X_val), len(X_test)
```



    # output
    (182535, 22720, 22891)



## Batch Normalization

Batch normalization normalizes the inputs within a mini-batch before passing them to the next layer. That is, for each input feature $$x_i$$, we subtract the batch mean and divide by the batch standard deviation. A small constant  $$\epsilon$$ is commonly added for maintaining numerical stability (to avoid zero division):

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

This standardization gives $$\hat{x}_i$$ a mean close to 0 and a standard deviation close to 1 over the batch. This may limit the model's capacity if left unchanged. Therefore, we introduce learnable parameters $$\gamma$$ (scale) and $$\beta$$ (shift) for flexibility:

$$
BN = \gamma \hat{x}_i + \beta
$$

Batch normalization is typically applied after the affine transformation ($$Wx + b$$) and before the non-linearity (e.g., ReLU):

$$
act = \phi(\textrm{BN}(Wx))
$$

Pay attention that we omitted $$b$$ when using batch normalization. In practice, the bias $$b$$ becomes redundant, because the shifting role is already handled by $$\beta$$. Recall that `PyTorch` has `bias=False` option  as well (e.g. in `nn.Conv2d()`).

Batch normalization improves convergence in optimization and has regularization effect. The original paper by Ioffe and Szegedyattributes this to reducing _internal covariate shift_ — i.e. the shift in the distribution of layer inputs during training as parameters in earlier layers change. But this intuition is challenged. You can read more about that in [d2l book chapter](https://d2l.ai/chapter_convolutional-modern/batch-norm.html#discussion) dedicated to batch normalization.


```python
########## PARAMETER SETUP ##########

def get_params(batch_norm=True):
  C = torch.randn((VOCAB_SIZE, EMBED_SIZE), requires_grad=True)

  W1 = torch.randn((BLOCK_SIZE * EMBED_SIZE, LAYER_SIZE), requires_grad = True)
  b1 = torch.zeros(LAYER_SIZE, requires_grad=True)

  W2 = torch.randn((LAYER_SIZE, VOCAB_SIZE), requires_grad = True)
  b2 = torch.zeros(VOCAB_SIZE, requires_grad=True)

  params = {'C': C, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

  if batch_norm:
    gamma = torch.ones((1, LAYER_SIZE), requires_grad=True)
    beta = torch.zeros((1, LAYER_SIZE), requires_grad=True)
    params['gamma'] = gamma
    params['beta'] = beta
    # we can add additional code for omitting b1 in case of using beta (BN bias)

  return params
```

### running_stats

In `PyTorch` we use `model.eval()` during inference to switch the model into evaluation mode. This is important because layers like dropout and batch normalization behave differently during training and evaluation.

During inference, normalization should be done using statistics over the whole dataset instead of mini-batches. Without `bn_stats` in the code below, the model would normalize using the current batch's mean and standard deviation, leading to inconsistent results depending on the batch.

The implemented `PyTorch` layers like [nn.BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) automatically calculate **running statistics** during training. These statistics include a running mean and a running variance for each feature channel, which are stored as non-learnable buffers inside the `BatchNorm` layer.

$$
\mu_{\text{running}} = \alpha \, \mu_{\text{batch}} + (1 - \alpha) \, \mu_{\text{running}}
$$

$$
\sigma^2_{\text{running}} = \alpha \, \sigma^2_{\text{batch}} + (1 - \alpha) \, \sigma^2_{\text{running}}
$$

In `BatchNorm`, $$\alpha$$ is defined as `momentum` which is a misnomer and has nothing to do with the momentum we had previously learned for optimization. Its values controls how quickly the `running_stats` adapt. If momentum is high, the running statistics update quickly based on new batches which can make them unstable and noisy if batches vary a lot. If it is low (by default it is set to `0.1`, but you may want to reduce it further depending on circumstances), the updates are smoother and slower, averaging batch statistics over time.

During evaluation `BatchNorm` uses the stored running mean and variance for normalization. This ensures deterministic behavior, regardless of the input batch. These buffers are automatically updated and used unless you disable tracking by setting `track_running_stats=False`.

A manual implementation of `running_stats` is demonstrated in Karpathy's video as well. In this notebook, we will only implemented the simpler `bn_stats`.


```python
########## FORWARD PASS ##########

@torch.no_grad() # applies "with torch.no_grad()" to the whole function
def get_bn_stats(X_train, params):
  emb = params['C'][X_train]
  out = emb.view(emb.shape[0], -1) @ params['W1'] + params['b1']
  mean, std = out.mean(0, keepdim=True), out.std(0, keepdim=True) + 1e-5
  return mean, std

def forward(X, params, batch_norm=False, bn_stats=None):
  emb = params['C'][X]
  out = emb.view(emb.shape[0], -1) @ params['W1'] + params['b1']

  if batch_norm:
    mean, std = bn_stats if bn_stats else (out.mean(0, keepdim=True), out.std(0, keepdim=True) + 1e-5)
    out = (out - mean) / std
    out = params['gamma'] * out + params['beta']

  act = torch.tanh(out)
  logits = act @ params['W2'] + params['b2']
  return logits
```


```python
########## TRAINING & INFERENCE ##########

def train(X, Y, params, num_epochs=100, lr=0.1, batch_size=None, batch_norm=False):
  for epoch in range(1, num_epochs+1):
    if batch_size:
      idx = torch.randint(0, X.size(0), (batch_size,))
      batch_X, batch_Y = X[idx], Y[idx]
    else:
      batch_X, batch_Y = X, Y

    logits = forward(batch_X, params, batch_norm)
    loss = F.cross_entropy(logits, batch_Y)

    for p in params.values():
      if p.grad is not None:
        p.grad.zero_()
    loss.backward()

    with torch.no_grad():
      for p in params.values():
        p.data -= lr * p.grad

    if epoch % (1000 if batch_size else 10) == 0:
      print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

@torch.no_grad()
def evaluate(X, Y, params, batch_norm=False, bn_stats=None):
  logits = forward(X, params, batch_norm, bn_stats)
  loss = F.cross_entropy(logits, Y)
  print(f"Loss: {loss.item():.4f}")
```


```python
########## TEST ##########

epochs = 10_000
lr = 0.01
batch_size = 32
batch_norm = True
init = True
```


```python
params = get_params(batch_norm=batch_norm)

# xavier for tanh, kaiming for relu
if init:
  nn.init.xavier_uniform_(params['W1'])
  nn.init.xavier_uniform_(params['W2'])
```


```python
train(X_train, Y_train, params, num_epochs=epochs, lr=lr, batch_size=batch_size, batch_norm=batch_norm)
```
    # output
    Epoch 1000, Loss: 2.7862
    Epoch 2000, Loss: 2.5897
    Epoch 3000, Loss: 2.3340
    Epoch 4000, Loss: 2.4985
    Epoch 5000, Loss: 2.7501
    Epoch 6000, Loss: 2.2715
    Epoch 7000, Loss: 2.6008
    Epoch 8000, Loss: 2.0696
    Epoch 9000, Loss: 2.3910
    Epoch 10000, Loss: 2.4777
    


```python
bn_stats = get_bn_stats(X_train, params) if batch_norm else None

print('Train and Validation losses:')
evaluate(X_train, Y_train, params, batch_norm=batch_norm, bn_stats=bn_stats)
evaluate(X_val, Y_val, params, batch_norm=batch_norm, bn_stats=bn_stats)
```
    # output
    Train and Validation losses:
    Loss: 2.3337
    Loss: 2.3345
    


```python
########## SAMPLING ##########

# minor changes to what we had previously for adapting to new code
def sample(params, n=10, batch_norm=False, bn_stats=None):
  names = []
  for _ in range(n):
    context = ['<START>'] * BLOCK_SIZE
    name = ''
    while True:
      X = torch.tensor([[stoi[c] for c in context]])
      logits = forward(X, params, batch_norm, bn_stats)
      probs = F.softmax(logits, dim=1)
      id = torch.multinomial(probs, num_samples=1).item()
      char = itos[id]
      if char == '<END>':
        break
      name += char
      context = context[1:] + [char]
    names.append(name)
  return names
```


```python
sample(params, batch_norm=batch_norm, bn_stats=bn_stats)
```

    # output
    ['khuc',
     'boka',
     'lyq',
     'rasyanrith',
     'onna',
     'helia',
     'brhaylanio',
     'boleiklak',
     'ekbnqron',
     'aren']



### Layer Normalization

A rule of thumb is that batch sizes between `50-100` generally work well for batch normalization: the batch is large enough to return reliable statistics but not so large that it causes memory issues or slows down training unnecessarily. Batch size of `32` is usually the lower bound where batch normalization still provides relatively stable estimates. Batch size of `128` is also effective if the hardware allows, and can produce even smoother estimates. Beyond that the benefit often diminishes.

If the batch size is very small due to memory limitations, batch normalization may lose its effectiveness. In such cases, it's better to consider alternatives like [Layer Normalization](https://arxiv.org/abs/1607.06450) which do not depend on the batch dimension.

Layer normalization normalizes across features for each individual sample, not across the batch and works well for _transformers_ where batch sizes may be small or variable. Basically, batch normalization depends on the batch, but layer normalization does not.

Furthermore, in fully connected layers, each feature is just a single number per sample, so batch normalization computes the mean and variance across the batch for each feature. Fully connected layers don't have spatial structure, so there's nothing to average across except the batch. In convolutional layers, each feature channel height and width and is a 2D map (hence, `nn.BatchNorm2d`), so batch normalization uses not just the batch dimension, but also all the spatial positions to compute statistics. This gives more stable estimates because there are more values per channel.

## Residual Block

**Residual Network (ResNet)** consists of repeated _residual blocks_, in the style of the VGGNet architecture. Each residual block consists of a _residual (skip/shortcut) connection_ . We will first see what it does and then will attempt to understand the reasoning behind this simple breakthrough idea.

### Implementation

{% assign img_caption = "Figure 8.6.2 of <a href='https://d2l.ai/chapter_convolutional-modern/resnet.html'>Dive into Deep Learning (Chapter 8)</a> by <a href='https://d2l.ai/'>d2l.ai</a> authors and contributors. Licensed under <a href='https://www.apache.org/licenses/LICENSE-2.0'>Apache 2.0</a>." %}

{% include figcaption.html src="https://d2l.ai/_images/residual-block.svg" alt="Residual Block" caption=img_caption inline=false %}

Hence, the idea of the residual connection is very simple. Before the second activation function, we add the previous input to the affine transformation. You can imagine the simplified code as below:


```python
def residual_block(X):
  act = torch.relu(X @ params['W1'] + params['b1'])
  out = act @ params['W2'] + params['b2']
  return torch.relu(out + X)
```

However, If we attempt to directly run the code above, we will see a shape mismatch, as our final layer returns a matrix of dimension `VOCAB_SIZE` which is not equal to the input dimension `BLOCK_SIZE * EMBED_SIZE`.

**Exercise:** Modifying the `forward` function by adding a residual connection.


```python
def forward(X, params, batch_norm=False, bn_stats=None, residual=True):
  emb = params['C'][X]
  out = emb.view(emb.shape[0], -1) @ params['W1'] + params['b1']

  if batch_norm:
    mean, std = bn_stats if bn_stats else (out.mean(0, keepdim=True), out.std(0, keepdim=True) + 1e-5)
    out = (out - mean) / std
    out = params['gamma'] * out + params['beta']

  act = torch.tanh(out + emb) if residual else torch.tanh(out)
  logits = act @ params['W2'] + params['b2']
  return logits
```


```python
X = params['C'][X_train].view(X_train.shape[0], -1)
X.shape
```

    # output
    torch.Size([182535, 30])



What to do? For demonstration purposes we will have to add another layer.

**Exercise (Advanced)**: Train a three layer model with batch normalization and residual connections.


```python
def get_params(batch_norm=True):
  C = torch.randn((VOCAB_SIZE, EMBED_SIZE), requires_grad=True)

  in_features = BLOCK_SIZE * EMBED_SIZE

  W1 = torch.randn((in_features, LAYER_SIZE), requires_grad = True)
  b1 = torch.zeros(LAYER_SIZE, requires_grad=True)

  W2 = torch.randn((LAYER_SIZE, in_features), requires_grad = True)
  b2 = torch.zeros(in_features, requires_grad=True)

  W3 = torch.randn((in_features, VOCAB_SIZE), requires_grad = True)
  b3 = torch.zeros(VOCAB_SIZE, requires_grad=True)

  params = {'C': C, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

  if batch_norm:
    gamma = torch.ones((1, LAYER_SIZE), requires_grad=True)
    beta = torch.zeros((1, LAYER_SIZE), requires_grad=True)
    params['gamma'] = gamma
    params['beta'] = beta

  return params
```


```python
def forward(X, params, batch_norm=False, bn_stats=None, residual=True):
  emb = params['C'][X].view(X.shape[0], -1)
  out = emb @ params['W1'] + params['b1']

  if batch_norm:
    mean, std = bn_stats if bn_stats else (out.mean(0, keepdim=True), out.std(0, keepdim=True) + 1e-5)
    out = (out - mean) / std
    out = params['gamma'] * out + params['beta']

  act = torch.relu(out)
  out2 = act @ params['W2'] + params['b2']

  if residual:
    out2 = out2 + emb

  logits = torch.tanh(out2) @ params['W3'] + params['b3']
  return logits
```


```python
params = get_params()
params.keys()
```


    # output
    dict_keys(['C', 'W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'gamma', 'beta'])




```python
# we are using relu in intermediate layer
if init:
  nn.init.kaiming_uniform_(params['W1'])
  nn.init.kaiming_uniform_(params['W2']);
```


```python
train(X_train, Y_train, params, num_epochs=epochs, lr=lr, batch_size=batch_size, batch_norm=batch_norm)
```
    # output
    Epoch 1000, Loss: 2.5227
    Epoch 2000, Loss: 2.9970
    Epoch 3000, Loss: 2.5845
    Epoch 4000, Loss: 2.3321
    Epoch 5000, Loss: 2.2630
    Epoch 6000, Loss: 2.5062
    Epoch 7000, Loss: 2.8853
    Epoch 8000, Loss: 2.3080
    Epoch 9000, Loss: 2.7023
    Epoch 10000, Loss: 2.8854
    


```python
bn_stats = get_bn_stats(X_train, params) if batch_norm else None

print('Train and Validation losses:')
evaluate(X_train, Y_train, params, batch_norm=batch_norm, bn_stats=bn_stats)
evaluate(X_val, Y_val, params, batch_norm=batch_norm, bn_stats=bn_stats)
```
    # output
    Train and Validation losses:
    Loss: 2.3690
    Loss: 2.3698
    

### Reasoning

As our model is implementing a single residual block, we don't see any performance improvement. However, similar to batch normalization, the advantages will be obvious in case of 50 layers or more, with repeated residual blocks. But why adding input of the layer to the second affine transformation boosts training?

Let's take any deep learning model. The types of functions this model can learn depend on its design (e.g. number of layers, activation functions, etc). All these possible functions we can denote as class $$\mathcal{F}$$. If we cannot learn a perfect function for our data, which is usually the case, we can at least try to appoximate this function as closely as possible by minimizing a loss. We may assume that a more powerful model can learn more types of functions and show better performance. But that's not always the case. To achieve a better performance than a simpler model, our model must be capable of learning not only more functions but also all the functions the simpler model can learn. Simply, the possible function class of the more powerful model should be a superclass of the simpler model's function class $$\mathcal{F} \subseteq \mathcal{F}'$$. If the $${F}'$$ isn't an expanded version of $$\mathcal{F}$$, the new model might actually learn a function that is farther from the truth, and even show worse performance.

Refer to the figure above, where our residual output is $$f(x) = g(x) + x$$. One advantage of residual blocks is their regularization effect. What if some activation nodes in our network are unnecessary and increase complexity or learn bad representations? Instead of learning weights and biases, our residual block can now learn an identity function $$f(x) = x$$ by simply setting that nodes parameters to zero. As a result, our inputs will propagate faster while ensuring that the learned functions are within the biggest function domain. Residual blocks not only act as a regularizer, but also, unlike, say, _dropout_ which stops input from propagating, allow the network to learn more functions by helping inputs to "jump over" (skip) the nodes. And it is very important that the function classes of the model with residual blocks is a superset of the same model without such blocks. Finally, along the way, it deals with the vanishing gradient problem by simply increasing the output of each layer. To sum up, residual connection allows the model to learn more complex functions, while allowing it to easily learn simpler ones, which tackles the vanishing gradient problem and has a regularizing effect.

## Residual Network for NLP in PyTorch

Originally, the complete Residual Network was developed for image classification tasks, winning _ImageNet_ competition. Each of its residual block consisted of two `3x3` convolutions (inspired  by _VGGNet_), both integrating batch normalization, followed by a skip connection. Even though, ResNet model relies on convolutional layer, the concept of residual connections has been adapted for NLP models as well. The infamous **Transformer** model, introduced in the paper titled [Attention is All You Need](https://arxiv.org/pdf/1706.03762) incorporates residual connections heavily in its design, which is very similar to ResNet.


```python
class ResidualBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
    self.fc1 = nn.Linear(in_features=EMBED_SIZE, out_features=LAYER_SIZE, bias=False)
    self.fc2 = nn.Linear(in_features=LAYER_SIZE, out_features=EMBED_SIZE, bias=False)
    self.fc3 = nn.Linear(in_features=EMBED_SIZE, out_features=VOCAB_SIZE, bias=True)
    self.bn1 = nn.LazyBatchNorm1d()
    self.bn2 = nn.LazyBatchNorm1d()
    nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')

  # nn.LazyBatchNorm1d in 3D input expects shape (batch, channels, length) = (B, C, T)
  # it normalizes across the batch and time (token, block) dimensions for each channel, independently
  # we need to move that dimension to the middle (axis 1) with transpose(1, 2)
  def forward(self, X):
    emb = self.emb(X)                     # (BATCH_SIZE, BLOCK_SIZE, EMBED_SIZE)
    out = self.fc1(emb).transpose(1, 2)   # (BATCH_SIZE, LAYER_SIZE, BLOCK_SIZE) for BatchNorm1d
    out = self.bn1(out).transpose(1, 2)   # back to our dimensions
    act = F.relu(out)
    out = self.fc2(act).transpose(1, 2)
    out = self.bn2(out).transpose(1, 2)
    out += emb                            # shortcut connection
    logits = self.fc3(out)                # (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)
    return logits
```


```python
model = ResidualBlock()
cel = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
```




    ResidualBlock(
      (emb): Embedding(28, 10)
      (fc1): Linear(in_features=10, out_features=100, bias=False)
      (fc2): Linear(in_features=100, out_features=10, bias=False)
      (fc3): Linear(in_features=10, out_features=28, bias=True)
      (bn1): LazyBatchNorm1d(0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn2): LazyBatchNorm1d(0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )




```python
num_epochs = 10_000
batch_size = 32

for epoch in range(1, num_epochs+1):
  model.train()
  idx = torch.randint(0, X_train.size(0), (batch_size,))
  batch_X, batch_Y = X_train[idx], Y_train[idx]
  optimizer.zero_grad()
  logits = model(batch_X)     # (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)
  logits = logits[:, -1, :]   # (BATCH_SIZE, VOCAB_SIZE)
  loss = cel(logits, batch_Y)
  loss.backward()
  optimizer.step()
  if epoch % 1000 == 0 or epoch == 1:
    print(f'Epoch {epoch}, Loss: {loss.item()}')
```
    # output
    Epoch 1, Loss: 3.6320574283599854
    Epoch 1000, Loss: 2.374105930328369
    Epoch 2000, Loss: 2.6409666538238525
    Epoch 3000, Loss: 2.6358656883239746
    Epoch 4000, Loss: 2.36672043800354
    Epoch 5000, Loss: 2.696502208709717
    Epoch 6000, Loss: 2.4992451667785645
    Epoch 7000, Loss: 2.413964033126831
    Epoch 8000, Loss: 2.83028507232666
    Epoch 9000, Loss: 2.3721745014190674
    Epoch 10000, Loss: 2.6832263469696045
    


```python
model.eval()
with torch.no_grad():
  logits_train = model(X_train)[:, -1, :]
  logits_val   = model(X_val)[:, -1, :]

  full_loss_train = cel(logits_train, Y_train)
  full_loss_val   = cel(logits_val, Y_val)

  print(f'Train loss: {full_loss_train.item()}')
  print(f'Validation loss: {full_loss_val.item()}')
```
    # output
    Train loss: 2.4901065826416016
    Validation loss: 2.4812421798706055


```python
# modifying code to suit our needs
def sample(model, n=10, block_size=3):
  model.eval()
  names = []
  for _ in range(n):
    context = ['<START>'] * block_size
    name = ''
    while True:
      idx = [stoi[c] for c in context]
      X = torch.tensor([idx], dtype=torch.long)
      with torch.no_grad():
        logits = model(X)[0, -1] # VOCAB_SIZE
      probs = F.softmax(logits, dim=0)
      idx_next = torch.multinomial(probs, num_samples=1).item()
      char = itos[idx_next]
      if char == '<END>':
        break
      name += char
      context = context[1:] + [char]
    names.append(name)
  return names
```

```python
sample(model)
```
    # output
    ['kelifo',
     'ja',
     'tha',
     'elarhncasoria',
     'ka',
     'voratte',
     'eniysh',
     'th',
     'kelld',
     'edm']