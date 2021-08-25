local numeric = {}
local h = 0.00001

-- Compute the numeric derivative of f : R -> R
function numeric.derivative(f)
  return function(x)
    return (f(x + h) - f(x - h))/(2*h)
  end
end

-- Take the gradient of a function f : R^n -> R
function numeric.gradient(f)
  return function(xs)
    local g = {}

    for i, x in ipairs(xs) do
      xs[i] = xs[i] - h
      local a = f(xs)
      xs[i] = xs[i] + h + h
      local b = f(xs)

      g[i] = (b - a) / (2*h)
    end
    return g
  end
end

function numeric.round(x, decimals)
  if type(x) == 'table' then
    local rounded = {}
    for i, v in ipairs(x) do
      rounded[i] = numeric.round(v)
    end
    return rounded
  end

  decimals = decimals or 0
  local y = x * (10^decimals)

  return math.floor(y + 0.5) / (10^decimals)
end

return numeric
