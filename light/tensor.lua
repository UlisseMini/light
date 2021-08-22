local utils = require('light.utils')
local Tensor = {
  do_grad = true, -- contextually limit autodiff computation
}
local meta = {}

local function is_tensor(t)
  return getmetatable(t) == meta
end

local function typecheck(data)
  if type(data) == 'table' then
    -- pytorch also doesn't allow nested tensors
    if is_tensor(data) then
      error('nested tensors are not allowed because of issues with autodiff')
    end

    for i, v in ipairs(data) do
      typecheck(v)
    end
  elseif type(data) ~= 'number' then
    error(("Tensor contains non number '%s'"):format(data))
  end
end

function meta:__index(k)
  if type(k) == 'number' then
    if self._type == 'number' then
      error(('cannot index a scalar tensor %s at %s'):format(self, k))
    end

    assert(k ~= 0, '0 is an invalid index [hint: indices start at 1]')
    if k > #self then
      return nil -- index out of bounds
    end

    if #self._stride == 1 then
      assert(self._stride[1] == 1)
      return self.storage[k + self._offset]
    else
      local offset = self._offset + self._stride[1] * (k - 1)
      local size = utils.slice(self._size, 2)
      local stride = utils.slice(self._stride, 2)
      return Tensor._new(self.storage, size, stride, offset)
    end
  else
    return Tensor[k]
  end
end

function meta:__newindex(k, v)
  if type(k) == 'number' then
    assert(self._type ~= 'number', 'cannot call __newindex on a scalar tensor')

    assert(type(v) == 'number', 'not supporting __newindex with tables yet')
    assert(#self._size == 1, ('not supporting __newindex on %s-dim tensors yet'):format(#self._size))

    if k > #self then
      error(('index out of bounds, len %s but attempt to set self[%s] = %s'):format(#self.data, k, v))
    end

    self.storage[self._offset + self._stride[1]*k] = v
  else
    rawset(self, k, v)
  end
end

function meta:__len()
  if self._type == 'number' then
    assert(#self.storage == 1, 'want length of 1 for scalar tensor, got ' .. tostring(#self.storage))
    error(('attempt to get len of 0-dim tensor %s'):format(self.storage[1]))
  end

  return self._size[1] or 0 -- also a hack to deal with {}
end


function Tensor.new(data, args)
  if is_tensor(data) then
    if args then
      error('Cannot make a tensor from a tensor with args because of ambiguity')
    else
      return data -- already a tensor
    end
  end

  typecheck(data)
  local storage = utils.flatten({data})
  local size = Tensor._size(data)
  local stride = Tensor._stride(size)
  local self = Tensor._new(storage, size, stride)

  if args and Tensor.do_grad then
    self.grad = nil               -- grad of this node, computed after calling backward
    self._parents = args._parents -- nodes that produced this node, eg. c = a + b, parents = {a,b}
    self._backward = args._backward
  end

  return self
end
setmetatable(Tensor, {__call = function(_, ...) return Tensor.new(...) end})

function Tensor._new(storage, size, stride, offset)
  assert(type(size) == 'table', 'want table size got ' .. type(size))
  assert(type(stride) == 'table', 'want table stride got ' .. type(stride))

  local self = {}
  self.storage = storage
  self._stride = stride
  self._size = size
  self._offset = offset or 0
  if #size == 0 then
    self._type = 'number'
  else
    -- tensor might confuse people, since 'number' is the same as lua type
    self._type = 'table'
  end

  setmetatable(self, meta)
  return self
end

function Tensor._stride(size)
  local stride = {}
  local dim = #size

  stride[dim] = 1
  for i=dim-1,1,-1 do
    stride[i] = size[i+1]*stride[i+1]
  end

  return stride
end


function Tensor:stride()
  return self._stride
end

function Tensor._size(t)
  local size = {}
  if type(t) == 'table' then
    size[1] = #t
    if type(t[1]) == 'table' then
      size = utils.concat(size, Tensor._size(t[1]))
    end
  end
  return size
end

function Tensor:size()
  return self._size
end

function Tensor:item()
  assert(self._type == 'number', 'cannot get item of a non scalar tensor')

  return self.storage[1 + self._offset]
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

-- TODO: return a view, add tests for mutation to ensure its a view
function Tensor:T()
  local res = {}

  for i=1,#self[1] do
    res[i] = {}
    for j=1,#self do
      res[i][j] = self[j][i]
    end
  end

  local t = Tensor(res)
  if self.grad then
    t.grad = self.grad:T()
  end
  t._parents = self._parents -- TODO: transpose?
  t._backward = self._backward
  return t
end

-- TODO: Implement
function Tensor:view(size)
  size = Tensor(size)
  local our_size = Tensor(self:size())

  local a, b = our_size:prod(), size:prod()
  if a ~= b then
    error(('size mismatch, tensor with %s items cannot be viewed as if it had %s items'):format(a, b))
  end

  -- See https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
  -- And https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch.Tensor.view

end

function Tensor.matmul(A, B)
  assert(A:size()[2] == B:size()[1], 'size mismatch')

  -- (n by m) * (m by p) = (n by p)
  local n, m, p = A:size()[1], A:size()[2], B:size()[2]
  local res = {}

  if p == nil then
    -- We got a vector, view it as a {m, 1} matrix.
    p = 1
    B = B:view({m, p})
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

meta.__add  = piecewiseOp(function(a,b) return a+b  end, function(a,b) return 1, 1 end)
meta.__sub  = piecewiseOp(function(a,b) return a-b  end, function(a,b) return 1, -1 end)
meta.__mul  = piecewiseOp(function(a,b) return a*b  end, function(a,b) return b, a end)
meta.__div  = piecewiseOp(function(a,b) return a/b  end, function(a,b) return 1/b, -a/b^2 end)
meta.__pow  = piecewiseOp(function(a,b) return a^b  end, function(a,b) return b*a^(b-1), math.log(a)*a^b end)
meta.__idiv = piecewiseOp(function(a,b) return a//b end)
meta.__mod  = piecewiseOp(function(a,b) return a%b  end)

-- **IMPORTANT:** Lua will only try __eq when the values being compared are *both tables.*
-- The result of the call is always converted to a boolean, so we can't return
-- a tensor then have .all() and .any() like numpy.
meta.__eq = function(a,b)
  if a._type ~= b._type then
    return false
  elseif a._type == 'number' and b._type == 'number' then
    return a:item() == b:item()
  end
  assert(a._type == 'table' and b._type == 'table')

  if #a ~= #b then
    return false
  end

  for i=1,#a do
    -- this recurses if a and b are tensors
    if a[i] ~= b[i] then
      return false
    end
  end
  return true
end

function meta:__tostring()
  return utils.pp(self)
end

return Tensor
