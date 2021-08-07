local Tensor = {}
local meta = {}

function meta:__tostring()
  local s = 'Tensor {\n'
  for i in ipairs(self) do
    s = s .. '\t' .. tostring(self[i]) .. ',\n'
  end
  s = s .. '}'
  return s
end

-- For method lookups, eg: mytensor:map(...)
function meta:__index(k)
  return Tensor[k]
end

function Tensor:new(o)
  assert(type(o) == 'table')

  setmetatable(o, meta)
  return o
end


function Tensor:map(fn)
  local res = Tensor:new({})
  for i,v in ipairs(self) do res[i] = fn(v) end
  return res
end

function Tensor:reduce(fn, acc)
  for _, curr in ipairs(self) do
    acc = fn(acc, curr)
  end
  return acc
end

function Tensor.piecewise(op, a, b)
  assert(#a == #b, ('#a (%d) != #b (%d)'):format(#a, #b))
  local res = Tensor:new({})
  for i=1,#a do
    res[i] = op(a[i], b[i])
  end
  return res
end

local piecewiseOp = function(op)
  return function(a,b)
    -- at most one of (a,b) is a number. if both were numbers we would never be called.
    if type(a) == 'number' then return b:map(function(v) return op(a,v) end) end
    if type(b) == 'number' then return a:map(function(v) return op(v,b) end) end

    -- both are tensors, preform op piecewise
    return Tensor.piecewise(op, a, b)
  end
end


-- See https://www.lua.org/manual/5.3/manual.html#2.4 for reference

meta.__add  = piecewiseOp(function(a,b) return a+b  end)
meta.__sub  = piecewiseOp(function(a,b) return a-b  end)
meta.__mul  = piecewiseOp(function(a,b) return a*b  end)
meta.__div  = piecewiseOp(function(a,b) return a/b  end)
meta.__idiv = piecewiseOp(function(a,b) return a//b end)
meta.__mod  = piecewiseOp(function(a,b) return a%b  end)

-- Lua will only try __eq when the values being compared are both tables.
-- The result of the call is always converted to a boolean, so we can't return
-- a tensor then have .all() and .any() like numpy.
meta.__eq = function(a,b)
  -- TODO: Use shapes once I add them
  if #a ~= #b then return false end

  local t = piecewiseOp(function(a,b) return a==b end)(a,b)
  return t:reduce(function(a,b) return a and b end, true)
end

return Tensor
