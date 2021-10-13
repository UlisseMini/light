import numpy as np
import timeit

images = np.random.random(size=(50000, 784))
weights = np.random.random(size=(784, 10))

number = 10
print('took {:.3f}s'.format(timeit.timeit(lambda: images @ weights, number=number) / number))
