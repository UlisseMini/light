local light = require('light')
local T = light.Tensor
local V = light.Value
math.randomseed(42)

local Net = {}
Net.__index = Net

function Net.new()
  local self = {}
  setmetatable(self, Net)
  return self
end

-- Init a neural network from weights and biases, where
-- weights is a list of matrices (for each layer) and biases
-- is a list of vectors for each layer.
function Net:init(layers, weights, biases)
  assert(#weights == #biases)

  self.weights = weights:map(V)
  self.biases = biases:map(V)
  self.layers = layers
end

-- Init a network randomly using math.random
function Net:random_init(layers)
  local weights = T{}
  local biases = T{}
  for i=1,#layers-1 do
    table.insert(biases, T.all({layers[i+1]}, math.random))

    table.insert(weights, T.all({layers[i+1], layers[i]}, math.random))
  end
  self:init(layers, weights, biases)
end


function Net:forward(a)
  for i=1,#self.weights do
    local W = self.weights[i]
    local b = self.biases[i]
    a = W:matmul(a) + b
  end
  return a
end

function Net:parameters()
  local i = 0
  local reading_weights = true
  return function()
    i = i + 1

    if reading_weights then
      if i > #self.weights then
        reading_weights = false
      end
      return self.weights[i]
    else
      return self.biases[i]
    end
  end
end

local net = Net.new()
net:random_init({3, 2, 5, 4})

local inputs = T{1,2,3}
local lr = 0.001

for epoch=1,1000 do
  local outputs = net:forward(inputs)
  local loss = outputs:dot(outputs)
  loss:backward()

  print('loss', loss.data)

  light.no_grad(function()
    net.weights:map_(function(w) return w - lr * w.grad end)
    net.biases:map_(function(b)  return b - lr * b.grad end)
  end)
end
