local images = {}

local pixels = 28*28
local classes = 10
local batchsize = 50000

print(('generating %s images of size %s (%s numbers)'):format(batchsize, pixels, batchsize*pixels))

for i=1,batchsize do
  images[i] = {}
  for j=1,pixels do
    images[i][j] = math.random()
  end
end

print(('generating %s weights'):format(classes*pixels))
local weights = {}

for i=1,classes do
  weights[i] = {}
  for j=1,pixels do
    weights[i][j] = math.random()
  end
end

print('evaluating model')
local start = os.clock()

local results = {}
for i, image in ipairs(images) do
  local scores = {}
  for j=1,classes do
    local s = 0
    local weights_j = weights[j]
    for k=1,pixels do
      s = s + weights_j[k] * image[k]
    end
    scores[j] = s
  end
end

local took = os.clock() - start
print(('took %.3fs'):format(took))
