from jax.lax import conv
import jax.random as jrd
import jax
print(jax.devices())

rng = jrd.PRNGKey(0)
x = jrd.truncated_normal(rng, -1, 1, (1, 1, 28, 28))
kernel = jrd.truncated_normal(rng, -1, 1, (2, 1, 3, 3))
y = conv(x, kernel, window_strides=(1, 1), padding='SAME')
