local utils = require('light.utils')
local F = require('light.F')

local Tensor = {}

Tensor.__index = Tensor

function Tensor.new(t)
  if type(t) ~= 'table' then
    error('cannot create tensor from ' .. type(t))
  end

  setmetatable(t, Tensor)
  return t
end
setmetatable(Tensor, {__call = function(_, ...) return Tensor.new(...) end})

function Tensor.size(t)
  local size = {}
  if type(t) == 'table' then
    size[1] = #t
    local subSize = Tensor.size(t[1])
    -- check all children have size subSize
    for i=2,#t do
      local childSize = Tensor.size(t[i])
      if not utils.eq(subSize, childSize) then
        error(('want subsize %s but got %s in position %s of %s'):format(
            utils.pp(subSize), utils.pp(childSize), i, utils.pp(t)
          ))
      end
    end

    size = utils.concat(size, subSize)
  end
  return size
end

-- since Tensor.new doesn't allow creating a tensor from tensors, we use a table
-- function for recursion, then convert to a tensor in Tensor.all
-- TODO: Refactor (no longer disallowing nested tensors)
local function tableAll(size, const)
  local t = {}
  if size == nil or #size == 0 then
    return const
  end

  for i=1,size[1] do
    t[i] = tableAll(utils.slice(size, 2), const)
  end

  return t
end

function Tensor.all(size, const)
  return Tensor(tableAll(size, const))
end

function Tensor.ones(size) return Tensor.all(size, 1) end
function Tensor.zeros(size) return Tensor.all(size, 0) end

--------------------- Tensor ops --------------------- 

function Tensor.matmul(A, B)
  assert(A:size()[2] == B:size()[1], 'size mismatch')

  -- (n by m) * (m by p) = (n by p)
  local n, m, p = A:size()[1], A:size()[2], B:size()[2]
  local res = {}

  if p == nil then
    p = 1
    local x = B
    for i=1,n do
      local s = 0
      for k=1, m do s = s + A[i][k] * x[k] end
      res[i] = s
    end
  else
    -- matrix matrix product
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
  end

  return Tensor(res)
end

function Tensor:dot(other)
  assert(#self == #other, 'vectors are of different lengths')

  return (self*other):sum()
end

function Tensor:reduce(fn, acc)
  utils.walk(self, function(cur) acc = fn(acc, cur) end)
  return acc
end

function Tensor:sum()  return self:reduce(F.add, 0) end
function Tensor:prod() return self:reduce(F.mul, 1) end

function Tensor:map(fn)
  local res = {}
  for _, v in ipairs(self) do
    if utils.number(v) then
      table.insert(res, fn(v))
    else
      table.insert(res, Tensor.map(v, fn))
    end
  end
  return Tensor(res)
end

function Tensor.piecewise(op, a, b)
  assert(#a == #b, ('#a (%d) != #b (%d)'):format(#a, #b))
  local res = {}
  for i=1,#a do
    res[i] = op(a[i], b[i])
  end
  return Tensor(res)
end

local piecewiseOp = function(op)
  return function(a, b)
    local na, nb = utils.number(a), utils.number(b)

    if na ~= nil and nb ~= nil then
      return op(na, nb)
    elseif na ~= nil then
      return b:map(function(v) return op(na, v) end)
    elseif nb ~= nil then
      return a:map(function(v) return op(v, nb) end)
    else
      -- both are tensors, preform op piecewise
      return Tensor.piecewise(op, a, b)
    end
  end
end

-- See https://www.lua.org/manual/5.3/manual.html#2.4 for reference

Tensor.__add  = piecewiseOp(F.add)
Tensor.__sub  = piecewiseOp(F.sub)
Tensor.__mul  = piecewiseOp(F.mul)
Tensor.__div  = piecewiseOp(F.div)
Tensor.__pow  = piecewiseOp(F.pow)

-- **IMPORTANT:** Lua will only try __eq when the values being compared are *both tables.*
-- The result of the call is always converted to a boolean, so we can't return
-- a tensor then have .all() and .any() like numpy.
function Tensor.__eq(a, b)
  return utils.eq(a, b)
end

function Tensor:__tostring()
  return utils.pp(self)
end

return Tensor
