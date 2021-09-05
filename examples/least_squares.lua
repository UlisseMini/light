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
  local x = T.zeros(b:size())
  -- enable gradient computation by turning every element of x into a Value,
  -- light currently does autodiff on the level of individual values.
  x = x:map(V)

  for i=1,1000 do
    local loss = cost(x)
    print(('loss %.4f\tx = %s'):format(loss.data, x:map(V.get.data)))
    -- in light :zero_grad is called automatically, if you want to accumulate gradients
    -- you call :backward_no_zero().
    loss:backward()

    -- Mapping V.get.grad grabs .grad from each Value in the x Tensor
    local grad = x:map(V.get.grad)
    -- Use no_grad so we don't backprop to old x, if you don't do this
    -- the code still works, but its slow and gets slower each iteration
    light.no_grad(function()
      x = x - lr * grad
    end)
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


