local utils = require('light.utils')

local Tensor = {
  do_grad = true, -- contextually limit autodiff computation
}

function Tensor:__index(k)
  if type(k) == 'number' then
    if type(self.data) == 'number' then
      error(('cannot index a scalar tensor %s at %s'):format(self.data, k))
    end

    return self.data[k]
  else
    return Tensor[k]
  end
end

function Tensor:__newindex(k, v)
  if type(k) == 'number' then
    assert(type(self.data) ~= 'number', 'cannot call __newindex on a scalar tensor')
    assert(type(v) == 'number', 'not supporting __newindex with tables or tensors yet')

    -- this should do .all when implemented (like numpy)
    assert(#self._size == 1, ('not supporting __newindex on %s-dim tensors yet'):format(#self._size))

    if k > #self then
      error(('index out of bounds, len %s but attempt to set self[%s] = %s'):format(#self.data, k, v))
    end

    self.data[k] = v
  else
    rawset(self, k, v)
  end
end

function Tensor:item()
  assert(type(self.data) == 'number')
  return self.data
end

function Tensor:__len()
  if type(self.data) == 'number' then
    error(('attempt to get len of 0-dim tensor %s'):format(self.data))
  end

  return self._size[1] or 0 -- also a hack to deal with {}
end

local function isinstance(t, m)
  return getmetatable(t) == m
end

local function typecheck(data)
  if type(data) == 'table' then
    -- pytorch also doesn't allow nested tensors
    if isinstance(data, Tensor) then
      error('nested tensors are not allowed because of issues with autodiff')
    end

    for i, v in ipairs(data) do
      typecheck(v)
    end
  elseif type(data) ~= 'number' then
    error(("Tensor contains non number '%s'"):format(data))
  end
end



function Tensor.new(data, args)
  if isinstance(data, Tensor) then
    if args then
      error('Cannot make a tensor from a tensor with args because of ambiguity')
    else
      return data -- already a tensor
    end
  end

  typecheck(data)
  local self = {}
  self._size = Tensor._size(data)
  self.data = data
  setmetatable(self, Tensor)

  if args and Tensor.do_grad then
    self.grad = nil               -- grad of this node, computed after calling backward
    self._parents = args._parents -- nodes that produced this node, eg. c = a + b, parents = {a,b}
    self._backward = args._backward
  end

  return self
end
setmetatable(Tensor, {__call = function(_, ...) return Tensor.new(...) end})

function Tensor._size(t)
  local size = {}
  if type(t) == 'table' then
    size[1] = #t
    local subSize = Tensor._size(t[1])
    -- check all children have size subSize
    for i=2,#t do
      local childSize = Tensor._size(t[i])
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

function Tensor:size()
  return self._size
end

-- since Tensor.new doesn't allow creating a tensor from tensors, we use a table
-- function for recursion, then convert to a tensor in Tensor.all
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

---------------------- Autodiff ---------------------- 

function Tensor.no_grad(fn)
  local old = Tensor.do_grad
  Tensor.do_grad = false
  return utils.finally(fn, function() Tensor.do_grad = old end)
end

function Tensor.no_grad_f(fn)
  return function(...)
    local args = table.pack(...)
    return Tensor.no_grad(function() fn(table.unpack(args)) end)
  end
end

function Tensor:backward()
  self.grad = Tensor.ones(self:size())

  local stack = {self}
  while #stack > 0 do
    for i=1, #stack do
      local node = table.remove(stack, 1)

      -- Go backward one layer deeper
      node:_backward(table.unpack(node._parents))

      -- We computed the gradients of node's parents, now we add its parents to the stack
      -- (but only its parents that are nonconstant, ie. parents that have parents)
      for i, p in ipairs(node._parents) do
        if p._parents then
          table.insert(stack, p)
        end
      end
    end
  end
end
-- don't compute gradients in backward
Tensor.backward = Tensor.no_grad_f(Tensor.backward)



--------------------- Tensor ops --------------------- 

function Tensor.matmul(A, B)
  assert(A:size()[2] == B:size()[1], 'size mismatch')

  -- (n by m) * (m by p) = (n by p)
  local n, m, p = A:size()[1], A:size()[2], B:size()[2]
  local res = {}

  if p == nil then
    p = 1
    -- matrix vector multiplication
    -- TODO: make cleaner with einsum or something?
    -- Separate gradient computation from functions so I can use dot() without guilt

  end

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

  return Tensor(res, {_parents = {A,B}, _backward = function(C,A,B)
    -- see https://cs231n.github.io/optimization-2/ for the derivation of the
    -- matrix multiplication derivative
    A.grad = (A.grad or 0) + C.grad:matmul(B:T())
    B.grad = (B.grad or 0) + A:T():matmul(C.grad)
  end})
end

function Tensor:dot(other)
  assert(#self == #other, 'vectors are of different lengths')

  return (self*other):sum()
end

function Tensor:sum()
  local s = 0
  for _, v in ipairs(self) do s = s + v end
  return Tensor(s, {
      _parents = {self},
      _backward = function(c, a)
        a.grad = (a.grad or 0) + Tensor.ones(a:size()) * c.grad
      end
    })
end

function Tensor:prod()
  local ret = 1
  for _, v in ipairs(self) do ret = ret * v end

  return Tensor(ret)
end

function Tensor:map(fn)
  local res = {}
  for _, v in ipairs(self) do
    table.insert(res, fn(v))
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

local piecewiseOp = function(op, derivs)
  local function backward(c, a, b)
    local da, db = derivs(a, b)
    a.grad = (a.grad or 0) + da * c.grad
    b.grad = (b.grad or 0) + db * c.grad
  end

  local forward = function(a, b)
    local ret

    local na, nb = utils.number(a), utils.number(b)

    if na ~= nil and nb ~= nil then
      ret = Tensor(op(na, nb))
    elseif na ~= nil then
      ret = b:map(function(v) return op(na, v) end)
    elseif nb ~= nil then
      ret = a:map(function(v) return op(v, nb) end)
    else
      -- both are tensors, preform op piecewise
      ret = Tensor.piecewise(op, a, b)
    end

    -- todo: respect a.no_grad and b.no_grad, maybe factor the checking logic out to __newindex?
    if Tensor.do_grad then
      ret._parents = {Tensor(a), Tensor(b)}
      ret._backward = backward
    end

    return ret
  end

  -- This table is here for testing, otherwise we could just return forward.
  local self = {forward = forward, op = op, derivs = derivs}
  setmetatable(self, {__call = function(_, ...) return forward(...) end})

  return self
end


-- See https://www.lua.org/manual/5.3/manual.html#2.4 for reference

Tensor.__add  = piecewiseOp(function(a,b) return a+b  end, function(a,b) return 1, 1 end)
Tensor.__sub  = piecewiseOp(function(a,b) return a-b  end, function(a,b) return 1, -1 end)
Tensor.__mul  = piecewiseOp(function(a,b) return a*b  end, function(a,b) return b, a end)
Tensor.__div  = piecewiseOp(function(a,b) return a/b  end, function(a,b) return 1/b, -a/b^2 end)
Tensor.__pow  = piecewiseOp(function(a,b) return a^b  end, function(a,b) return b*a^(b-1), math.log(a)*a^b end)
Tensor.__idiv = piecewiseOp(function(a,b) return a//b end)
Tensor.__mod  = piecewiseOp(function(a,b) return a%b  end)

-- **IMPORTANT:** Lua will only try __eq when the values being compared are *both tables.*
-- The result of the call is always converted to a boolean, so we can't return
-- a tensor then have .all() and .any() like numpy.
Tensor.__eq = function(a, b)
  return utils.eq(a.data, b.data)
end

function Tensor:__tostring()
  return utils.pp(self)
end


local light = {}
light.utils = utils
light.numeric = require('light.numeric')
light.Tensor = Tensor

return light
