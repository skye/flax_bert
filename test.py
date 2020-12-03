# General imports.
import os
import time
import jax
import jax.profiler
server = jax.profiler.start_server(9999)
print('started profiler')

y = 9
for _ in range(200):
  for x in range(300000):
    y = x * 2

