local light = require('light')
local T = light.Tensor
local V = light.Value
math.randomseed(42)
-- math.randomseed(1/os.clock())

local function rand(a, b)
  return math.random()*(b-a) + a
end

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

  self.layers = layers
  -- map V to enable gradient tracking for network parameters
  self.weights = weights:map(V)
  self.biases = biases:map(V)
end

-- Init a network randomly using math.random
function Net:random_init(layers)
  local weights = T{}
  local biases = T{}
  for i=1,#layers-1 do
    table.insert(biases, T.all({layers[i+1]}, function() return 0 end))
    table.insert(weights, T.all({layers[i+1], layers[i]}, function() return rand(-1,1) end))
  end
  self:init(layers, weights, biases)
end

function Net:forward(a)
  for i=1,#self.weights do
    local W = self.weights[i]
    local b = self.biases[i]
    a = (W:matmul(a) + b)

    -- Relu every layer except the last
    if i ~= #self.weights then
      a:map_(V.relu)
    end
  end
  return a
end

local net = Net.new()

local n_inputs = 2
local n_outputs = 1

-- The nerual network will learn to approximate this function
local function f(x)
  return x:sum()
end

local train = {}
for i=1,100 do
  local x = T.all({n_inputs}, function() return rand(-10, 10) end)
  local y = f(x)
  table.insert(train, {x, y})
end

local test = {}
for i=1,100 do
  local x = T.all({n_inputs}, function() return rand(20, 30) end)
  local y = f(x)
  table.insert(test, {x, y})
end

net:random_init({n_inputs, 2, n_outputs})

local function compute_loss(net, data)
  local loss = 0

  for i=1,#data do
    local input = data[i][1]
    local label = data[i][2]

    local output = net:forward(input)
    local err = output - label
    loss = loss + err:dot(err):sqrt()
  end
  loss = loss / #data
  return loss
end

local epochs = 100
local lr
for epoch=1,epochs do
  lr = 1 / epoch^2 -- lr decay

  local train_loss = compute_loss(net, train)
  train_loss:backward()
  local test_loss = light.no_grad(function() return compute_loss(net, test) end)
  print(('[%s]\ttrain loss %.6f test loss %.6f'):format(epoch, train_loss.data, test_loss.data))

  light.no_grad(function()
    net.weights:map_(function(w) return w - lr * w.grad end)
    net.biases:map_(function(b)  return b - lr * b.grad end)
  end)
end

print('Net params:')
print(net.weights:map(V.get.data))
print(net.biases:map(V.get.data))
