local pixels = 28*28
local classes = 10
local batchsize = 50000

print(('generating %s images of size %s (%s numbers)'):format(batchsize, pixels, batchsize*pixels))

local images = {}
for i=1,batchsize do
  images[i] = {}
  for j=1,pixels do
    images[i][j] = math.random()
  end
end

print(('generating %s weights'):format(classes*pixels))
local weights = {}

for i=1,pixels do
  weights[i] = {}
  for j=1,classes do
    weights[i][j] = math.random()
  end
end

local function matmul(A, B)
  local n, m, p = #A, #A[1], #B[1]

  local res = {}
  for i=1,n do
    res[i] = {}
    for j=1,p do
      local s = 0
      local Ai = A[i]
      for k=1,m do
        s = s + Ai[k] * B[k][j]
      end
      res[i][j] = s
    end
  end
  return res
end

print('evaluating model')
local start = os.clock()

local scores = matmul(images, weights)

local took = os.clock() - start
print(('took %.3fs'):format(took))
