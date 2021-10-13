local light = require('light')
local Tensor = light.Tensor


print('generating images and weights...')
local images = Tensor.all({50000, 784}, math.random)
local weights = Tensor.all({784, 10}, math.random)

print('evaluating model...')

local start = os.clock()
images:matmul(weights)
local took = os.clock() - start

print(('took %.3fs'):format(took))
