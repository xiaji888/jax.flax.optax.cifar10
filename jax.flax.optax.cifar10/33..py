import jax
import jax.numpy as jnp
import flax.nnx as nn
from flax import nnx
import optax
import numpy as np
import pickle
import os

class CIFAR10:
    def __init__(self, train):
        self.data, self.labels = self.load_data(train)
    
    def load_data(self, train):      
        data, labels = [], []
        for f in files:
            with open(os.path.join(path, f), 'rb') as file:
                batch = pickle.load(file, encoding='latin1')
                data.append(batch['data'])
                labels.extend(batch['labels'])
        
        data = np.vstack(data).reshape(-1, 32, 32, 3) / 255.0
        return data, np.array(labels)
    
    def get_batches(self, batch_size=128):
        indices = np.random.permutation(len(self.data))
        for i in range(0, len(self.data), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield {
                'image': jnp.array(self.data[batch_indices]),
                'label': jnp.array(self.labels[batch_indices])
            }
            
class CNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv(32, (3, 3))
        self.conv2 = nn.Conv(64, (3, 3))
        self.linear = nn.Linear(10)
    
    def __call__(self, x):
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = jnp.mean(x, axis=(1, 2))
        return self.linear(x)
model = CNN()
graphdef, params = nnx.split(model)
tx = optax.adam(0.001)
opt_state = tx.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(params):
        model = nnx.merge(graphdef, params)
        logits = model(batch['image'])
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss
train = True
dataset = CIFAR10(train)
for epoch in range(10):
    total_loss, count = 0, 0
    for batch in dataset.get_batches(128):
        params, opt_state, loss = train_step(params, opt_state, batch)
        total_loss += loss
        count += 1
    print(f'Epoch {epoch+1}, Loss: {total_loss/count:.4f}')
