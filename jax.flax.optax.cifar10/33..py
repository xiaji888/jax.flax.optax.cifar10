import jax
import jax.numpy as jnp
import flax.nnx as nn
from flax import nnx
import optax
import tensorflow_datasets as tfds

class CNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv(32, kernel_size=(3, 3))
        self.conv2 = nn.Conv(64, kernel_size=(3, 3)) 
        self.linear = nn.Linear(10)
    
    def __call__(self, x):
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))  
        x = self.linear(x)
        return x

rng = jax.random.PRNGKey(0)
model = CNN()
graphdef, params = nnx.split(model)
tx = optax.adam(0.001)
opt_state = tx.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(params):
        model = nnx.merge(graphdef, params)
        logits = model(batch['image'] / 255.0)  
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']
        ).mean()
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    updates, new_opt_state = tx.update(grads, opt_state, params)
    
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

train_ds = tfds.load('cifar10', split='train')
train_ds = train_ds.batch(128).prefetch(1)

print("开始训练...")
for epoch in range(10):  
    losses = []
    for batch in tfds.as_numpy(train_ds):
        params, opt_state, loss = train_step(params, opt_state, batch)
        losses.append(loss)
    
    avg_loss = jnp.mean(jnp.array(losses))
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

print("训练完成！")  
