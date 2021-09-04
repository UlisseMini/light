local light = require('light')
local T, V = light.Tensor, light.Value

local function least_squares(A, b)
  -- Minimize ||Ax - b||^2
  local function cost(x)
    local diff = (A:matmul(x) - b)
    local mag_squared = diff:dot(diff)
    return mag_squared
  end

  local lr = 0.01
  local x = T.zeros(b:size()):map(V)

  for i=1,1000 do
    local loss = cost(x)
    print(('loss %.4f\tx = %s'):format(loss.data, x:map(V.get.data)))
    loss:backward()

    local grad = x:map(V.get.grad)
    x = x - lr * grad
  end

  return x
end


local A = T{
  {1,2},
  {3,4},
}

local b = T{1, 2}
local x_star = T{0, 0.5} -- solution of Ax = b

local x = least_squares(A, b)
print(('got %s want %s'):format(x:map(V.get.data), x_star))


