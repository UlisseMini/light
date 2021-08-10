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

function Tensor.new(t)
  assert(type(t) == 'table')
  setmetatable(t, meta)

  for i in ipairs(t) do
    -- convert tables to tensors recursively
    if type(t[i]) == 'table' then
      if getmetatable(t[i]) ~= meta then
        t[i] = Tensor(t[i])
      end
    elseif type(t[i]) ~= 'number' then
      error(("Tensor contains non number '%s' in position '%s'"):format(t[i], i))
    end
  end

  return t
end
setmetatable(Tensor, {__call = function(_, ...) return Tensor.new(...) end})

local function slice(t, a)
  local ret = {}
  for i=a,#t do
    ret[i-1] = t[i]
  end
  return ret
end

function Tensor.all(size, const)
  local t = {}
  if size == nil or #size == 0 then
    return const
  end

  for i=1,size[1] do
    t[i] = Tensor.all(slice(size, 2), const)
  end

  return Tensor(t)
end

function Tensor.ones(size) return Tensor.all(size, 1) end
function Tensor.zeros(size) return Tensor.all(size, 0) end

function Tensor:size()
  if type(self[1]) == 'table' then
    return Tensor({#self, table.unpack(self[1]:size())})
  else
    return Tensor({#self})
  end
end

function Tensor.matmul(A, B)
  assert(A:size()[2] == B:size()[1], 'size mismatch')
  -- (n by m) * (m by p) = (n by p)
  local n, m, p = A:size()[1], A:size()[2], B:size()[2]

  local res = {}

  -- this is ugly, but it's just saying (AB)ij is dot(row i of A, col j of B)
  for i=1,n do
    res[i] = {}
    for j=1,p do
      local s = 0
      for k=1,m do
        s = s + A[i][k] * B[k][j]
      end
      res[i][j] = s
    end
  end

  return Tensor(res)
end

function Tensor:dot(other)
  assert(#self == #other, 'vectors are of different lengths')

  return (self*other):sum()
end

function Tensor:sum()
  local s = 0
  for _, v in ipairs(self) do s = s + v end
  return s
end

function Tensor:map(fn)
  local res = Tensor({})
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
  local res = Tensor({})
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
