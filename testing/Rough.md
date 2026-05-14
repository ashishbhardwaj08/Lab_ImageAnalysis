## (a) CNN Architecture

Given:

- Input image size =$(1 \times 32 \times 32)$ (grayscale image)
- Convolution uses:
    - Kernel size = (3 \times 3)
    - Stride = 1
    - Padding = 1
- MaxPooling:
    - Pool size = (2 \times 2)
    - Stride = 2
- Number of classes = 6

---

### Architecture Diagram

$$
\text{Input} \rightarrow \text{Conv1} \rightarrow \text{ReLU} \rightarrow \text{MaxPool}
\rightarrow \text{Conv2} \rightarrow \text{ReLU} \rightarrow \text{MaxPool}
\rightarrow \text{Flatten} \rightarrow \text{Fully Connected} \rightarrow \text{Softmax}
$$

---

### Step-by-Step Shapes

#### Input Layer

$1 \times 32 \times 32$

---

### Layer 1: Convolution

- Filters = 16
- Kernel = (3 x 3)
- Padding = 1
- Stride = 1

Output size formula: $\frac{(W - F + 2(P))}{S} + 1$

Substituting values:

$\frac{(32 - 3 + 2(1))}{1} + 1 = 32$

Therefore:

16 \times 32 \times 32

---

### ReLU

No change in dimensions:
16 x 32 x 32

---

### MaxPooling 1

Pool size (2 x 2), stride 2.

16 x 16 x 16

---

### Layer 2: Convolution

- Filters = 32
- Kernel = (3 x 3)
- Padding = 1
- Stride = 1

Spatial dimensions remain same:

32 x 16 x 16

---

### ReLU

No change:

32 x 16 x 16

---

### MaxPooling 2

32 x 8 x 8

---

### Flatten Layer

Flatten all values into a vector:

32 x 8 x 8 = 2048

So flattened vector size:

2048

---

### Fully Connected Layer

Output neurons = 6

$2048 \rightarrow 6$

---

### Softmax Output

Final probability vector:

6

(one probability for each class)

---

# Final Architecture Table

| Layer           | Operation                    | Output Shape             |
| --------------- | ---------------------------- | ------------------------ |
| Input           | Grayscale Image              | (1 \times 32 \times 32)  |
| Conv1           | 16 filters (3\times3)        | (16 \times 32 \times 32) |
| ReLU            | Activation                   | (16 \times 32 \times 32) |
| MaxPool1        | (2\times2)                   | (16 \times 16 \times 16) |
| Conv2           | 32 filters (3\times3)        | (32 \times 16 \times 16) |
| ReLU            | Activation                   | (32 \times 16 \times 16) |
| MaxPool2        | (2\times2)                   | (32 \times 8 \times 8)   |
| Flatten         | Vector conversion            | (2048)                   |
| Fully Connected | Dense Layer                  | (6)                      |
| Softmax         | Classification probabilities | (6)                      |

---

# (b) Shapes of Parameters and Activations

## Parameter Shapes

---

### First Convolution Layer

Input channels = 1  
Filters = 16  
Kernel size = (3 x 3)

Weight tensor:

$W_1 \in \mathbb{R}^{16 \times 1 \times 3 \times 3}$

Bias vector:

$b_1 \in \mathbb{R}^{16}$

---

### Second Convolution Layer

Input channels = 16  
Filters = 32

Weight tensor:

$W_2 \in \mathbb{R}^{32 \times 16 \times 3 \times 3}$

Bias vector:

[
$b_2 \in \mathbb{R}^{32}$
]

---

### Fully Connected Layer

Input features = 2048  
Output classes = 6

Weight matrix:

$W_3 \in \mathbb{R}^{6 \times 2048}$

Bias vector:

$b_3 \in \mathbb{R}^{6}$

---

# Activation Shapes for Batch Size (B)

If batch size is (B):

---

### Input

(B,1,32,32)

---

### After Conv1

(B,16,32,32)

---

### After ReLU1

(B,16,32,32)

---

### After MaxPool1

(B,16,16,16)

---

### After Conv2

(B,32,16,16)

---

### After ReLU2

(B,32,16,16)

---

### After MaxPool2

(B,32,8,8)

---

### After Flatten

(B,2048)

---

### Fully Connected Output

(B,6)

---

### Softmax Output

(B,6)

---

# (c) Synthetic Dataset Generation (PyTorch)

```python
import torch
import numpy as np

# Number of samples
N = 6000

# Generate random grayscale images
X = torch.rand(N, 1, 32, 32)

# Random labels from 0 to 5
y = torch.randint(0, 6, (N,))

# One-hot encoding
y_onehot = torch.zeros(N, 6)
y_onehot[torch.arange(N), y] = 1

print("Image Shape:", X.shape)
print("Label Shape:", y_onehot.shape)
```

---

# (d) CNN Implementation Using PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc = nn.Linear(32 * 8 * 8, 6)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        output = F.softmax(x, dim=1)

        return output
```

---

# (e) Batch Gradient Descent Training

```python
import torch.optim as optim

# Create model
model = CNN()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.005)

# Training
epochs = 10

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(outputs, y)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")
```

---

# Important Notes

## ReLU Activation

ReLU introduces non-linearity:

$f(x)=\max(0,x)$

- Negative values become 0
- Positive values remain unchanged
- Helps prevent vanishing gradients

---

## Softmax Function

Softmax converts scores into probabilities:

$\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$

- Output probabilities sum to 1
- Highest probability gives predicted class

---

## Cross Entropy Loss

Loss function used:

$L = -\sum\_{i=1}^{n} y_i \log(\hat{y}\_i)$

- Measures difference between predicted and true probability distributions
- Lower loss indicates better prediction accuracy

---
