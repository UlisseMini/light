local light = require('light')
local Tensor = light.Tensor
local V = light.Value
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

function idxn.readfile(path, n)
  -- see http://yann.lecun.com/exdb/mnist/ for file format specification
  local file = assert(io.open(path, 'rb'))

  file:read(4) -- Skip magic number
  local dims = {}
  for i=1,n do
     local chunk = file:read(4)
     local dim = string.unpack('>i', chunk)
     table.insert(dims, dim)
  end
  dims[1] = 64

  local data = file:read('a')
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
  self.weights = Tensor.all({10, 28, 28}, function() return math.random()/(28*28) end)
end

function Linear.new()
  local self = {}
  setmetatable(self, Linear)
  self:init()
  return self
end

function Linear:forward(image)
  local image = Tensor(image) / 255

  local scores = {}
  for i=1,num_labels do
    scores[i] = self.weights[i]:dot(image)
  end
  return Tensor(scores)
end

local function softmax(logits)
  local t = logits:map(V.exp)
  return t / t:sum()
end


local function label_vec(label)
  local t = Tensor.zeros({num_labels})
  t[label+1] = 1
  return t
end

local function loss_fn(l, image, label)
  local logits = l:forward(image)
  local softmaxed = softmax(logits)

  local y = label_vec(label)
  local loss = -(y * softmaxed:map(V.log)):sum()
  return loss
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
      local y_hat = l:forward(data[i])
      if y_hat:argmax() == labels[i]+1 then
        correct = correct + 1
      end
    end
    return correct / #data
  end)
end

local l = Linear.new()
l.weights:map_(V)

for epoch=1, 100 do
  local loss = minibatch_loss(l, train_data, train_labels)
  print(('loss %.3f accuracy %.3f'):format(loss.data, accuracy(l, train_data, train_labels)))

  loss:backward()

  light.no_grad(function()
    l.weights:map_(function(w) return w - (0.3/epoch) * w.grad end)
  end)
end
