local light = require('light')
local Tensor = light.Tensor
local image_size = 28
local num_images = 5

local idxn = {}

function idxn.read(data, dims, dim, offset)
  local t = {}
  if #dims == dim then
    -- base case
    for i=1,dims[dim] do
      t[i] = data:byte(offset + i)
    end
  else
    -- recursive case

    -- compute the product of the dims, ie. (50, 28, 28) -> 28*28 for dim=1
    local prod = 1
    for i=dim+1, #dims do
      prod = prod * dims[i]
    end

    -- iterate over this dim building up the tensor
    for i=1,dims[dim] do
      t[i] = idxn.read(data, dims, dim+1, offset + prod * (i-1) + 1)
    end
  end

  return t
end


-- making stuff work with lua and luajit is a pain
if not bit then
  bit = {}
  bit.lshift = assert(load('return function(a,b) return a << b end'))()
end

-- string.unpack('>i', s) in pure lua so luajit is happy
function sunpack(s)
  local v = 0
  for i=1,#s do
    v = v + s:byte(i) * bit.lshift(1, 8*(#s - i))
  end
  return v
end

function idxn.readfile(path, n)
  -- see http://yann.lecun.com/exdb/mnist/ for file format specification
  local file = assert(io.open(path, 'rb'))

  file:read(4) -- Skip magic number
  local dims = {}
  for i=1,n do
     local chunk = file:read(4)
     local dim = sunpack(chunk)
     table.insert(dims, dim)
  end
  -- TODO: Train on the full dataset
  dims[1] = 1000

  local data = file:read('*a')
  assert(file:close())


  return idxn.read(data, dims, 1, 0)
end

local function imshow(img)
  for i=1,#img do
    for j=1,#img[1] do
      local pixel = img[i][j]
      if pixel > 50 then
        io.write('* ')
      else
        io.write('. ')
      end
    end
    print()
  end
end


local train_data = Tensor(idxn.readfile('examples/mnist/train-images-idx3-ubyte', 3))
local train_labels = Tensor(idxn.readfile('examples/mnist/train-labels-idx1-ubyte', 1))

local num_labels = 10

print('label', train_labels[1])
imshow(train_data[1])

math.randomseed(42)

local Linear = {}
Linear.__index = Linear

function Linear:init()
  self.weights = Tensor.all({10, 28*28}, function() return math.random()/(28*28) end)
end

function Linear.new()
  local self = {}
  setmetatable(self, Linear)
  self:init()
  return self
end

local function tovec(image)
  return Tensor(light.utils.flatten(Tensor(image) / 255))
end

function Linear:forward(x)
  local y_hat = self.weights:matmul(x)

  -- basically matrix multiplication
  -- local scores = {}
  -- for i=1,num_labels do
  --   scores[i] = self.weights[i]:dot(image)
  -- end
  return y_hat
end

local function label_vec(label)
  local t = Tensor.zeros({num_labels})
  t[label+1] = 1
  return t
end

local function loss_fn(l, image, label)
  local y_hat = l:forward(tovec(image))
  local y = label_vec(label)
  local loss = ((y_hat - y)^2):sum()
  return loss
end

local function loss_grad(l, image, label)
  local y = label_vec(label)
  local x = tovec(image)
  local y_hat = l:forward(tovec(image))

  -- compute outer(2*(y_hat - y), x)
  local outer = {}
  for i=1,#y do
    outer[i] = {}
    for j=1,#x do
      outer[i][j] = 2*(y_hat[i] - y[i]) * x[j]
    end
  end
  return Tensor(outer)
end

local function minibatch_loss_grad(l, data, labels)
  local all = 0
  for i=1,#data do
    all = all + loss_grad(l, data[i], labels[i])
  end
  return all / #data
end

local function minibatch_loss(l, data, labels)
  assert(#data == #labels)

  local all_loss = 0
  for i=1,#data do
    all_loss = all_loss + loss_fn(l, data[i], labels[i])
  end
  return all_loss / #data
end

local function accuracy(l, data, labels)
  return light.no_grad(function()
    local correct = 0
    for i=1,#data do
      local y_hat = l:forward(tovec(data[i]))
      if y_hat:argmax() == labels[i]+1 then
        correct = correct + 1
      end
    end
    return correct / #data
  end)
end

local l = Linear.new()

-- local profiler = require('profiler')
-- profiler.start()

for epoch=1, 10 do
  local loss = minibatch_loss(l, train_data, train_labels)
  local maxw = l.weights:max()
  print(('[epoch %s] loss %.3f accuracy %.3f maxw %.3f'):format(epoch, loss, accuracy(l, train_data, train_labels), maxw))
  local grad = minibatch_loss_grad(l, train_data, train_labels)
  grad = grad / (grad*grad):sum()
  l.weights = l.weights - 0.1*grad

  -- light.no_grad(function()
  --   l.weights:map_(function(w) return w - (0.3/epoch) * w.grad end)
  -- end)
end

-- profiler.stop()
-- profiler.report('mnist.log')

