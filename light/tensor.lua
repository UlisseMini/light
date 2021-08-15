local Tensor = {
  do_grad = true, -- contextually limit autodiff computation
}
local meta = {}

local function is_tensor(t)
  return getmetatable(t) == meta
end

local function typecheck(data)
  if type(data) == 'table' then
    -- pytorch doesn't allow nested tensors: torch.tensor([tensor([1,2]), tensor([3,4])])
    -- gives an error. I think this is because of autodiff difficulties,
    -- returning a view of a tensor seems the correct approach.
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

function meta:__tostring()
  if type(self.data) == 'number' then
    return 'Tensor(' .. tostring(self.data) .. ')'
  end

  local s = 'Tensor {\n'
  for i in ipairs(self) do
    s = s .. '\t' .. tostring(self[i]) .. ',\n'
  end
  s = s .. '}'
  return s
end

function meta:__index(k)
  local x = Tensor[k]
  if x == nil and type(k) == 'number' then
    x = self.data[k]
  end
  if type(x) == 'table' then return Tensor(x) end
  return x
end

function meta:__newindex(k, v)
  if type(k) == 'number' then
    -- TODO: allow adding tensors via indexing, and add fancy indexing
    typecheck(v)
    self.data[k] = v
  else
    rawset(self, k, v)
  end
end

function meta:__len()
  return #self.data
end

function Tensor.new(data, args)
  -- do nothing if data is already a tensor
  -- TODO: will this cause issues if args is populated?
  if is_tensor(data) then
    return data
  end

  local self = {}

  typecheck(data)
  self.data = data

  -- args.no_grad = true disables recording of gradient information
  if args and not args.no_grad and Tensor.do_grad then
    self.grad = nil              -- grad of this node, computed after calling backward
    self._parents = args.parents -- nodes that produced this node, eg. c = a + b, parents = {a,b}
    self._backward = args.backward
  end

  setmetatable(self, meta)
  return self
end
setmetatable(Tensor, {__call = function(_, ...) return Tensor.new(...) end})

local function slice(t, a)
  local ret = {}
  for i=a,#t do
    ret[i - a + 1] = t[i]
  end
  return ret
end

local function tableAll(size, const)
  local t = {}
  if size == nil or #size == 0 then
    return const
  end

  for i=1,size[1] do
    t[i] = tableAll(slice(size, 2), const)
  end

  return t
end

function Tensor.all(size, const)
  return Tensor(tableAll(size, const))
end

function Tensor.ones(size) return Tensor.all(size, 1) end
function Tensor.zeros(size) return Tensor.all(size, 0) end

function Tensor:size()
  -- I have to use .data since self is always a table. I could overwrite
  -- __type but that seems like a bad idea
  local t = {}
  if type(self.data) == 'table' then
    t[1] = #self.data
    if type(self.data[1]) == 'table' then
      t = {table.unpack(t), table.unpack(self[1]:size())}
    end
  end

  return Tensor(t)
end


---------------------- Autodiff ---------------------- 

-- todo: put this in utils?
local function finally(fn, cleanup)
  local ret = table.pack(xpcall(fn, debug.traceback))
  cleanup()

  local status, err = table.unpack(ret)
  if not status then error(err) end

  table.remove(ret, 1)
  return table.unpack(ret)
end

function Tensor.no_grad(fn)
  local old = Tensor.do_grad
  Tensor.do_grad = false
  return finally(fn, function() Tensor.do_grad = old end)
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

      assert(#node._parents == 2, 'currently only supporting ops between two vars')
      local a, b = table.unpack(node._parents)
      local da, db = node._backward(a, b)

      a.grad = (a.grad or 0) + da * node.grad
      b.grad = (b.grad or 0) + db * node.grad

      if a._parents then table.insert(stack, a) end
      if b._parents then table.insert(stack, b) end
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

  return Tensor((self*other):sum(),
    {
      parents = {self, other},
      backward = function(a, b)
        return b, a
      end
    })
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

local function number(t)
  if type(t) == 'number' then
    return t
  elseif type(t) == 'table' and type(t.data) == 'number' then
    return t.data
  else
    return nil
  end
end

function Tensor.piecewise(op, a, b)
  assert(#a == #b, ('#a (%d) != #b (%d)'):format(#a, #b))
  local res = {}
  for i=1,#a do
    res[i] = op(a[i], b[i])
  end
  return Tensor(res)
end

local piecewiseOp = function(op, backward)
  local forward = function(a, b)
    local ret

    local na, nb = number(a), number(b)

    if na ~= nil and nb ~= nil then
      ret = Tensor(op(na, nb))
    elseif na ~= nil then
      ret = b:map(function(v) return op(na,v) end)
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

  local self = {forward = forward, backward = backward, op = op}
  setmetatable(self, {__call = function(_, ...) return forward(...) end})

  return self
end


-- See https://www.lua.org/manual/5.3/manual.html#2.4 for reference

meta.__add  = piecewiseOp(function(a,b) return a+b  end, function(a,b) return 1, 1 end)
meta.__sub  = piecewiseOp(function(a,b) return a-b  end, function(a,b) return 1, -1 end)
meta.__mul  = piecewiseOp(function(a,b) return a*b  end, function(a,b) return b, a end)
meta.__div  = piecewiseOp(function(a,b) return a/b  end, function(a,b) return 1/b, -a/b^2 end)
meta.__idiv = piecewiseOp(function(a,b) return a//b end)
meta.__mod  = piecewiseOp(function(a,b) return a%b  end)

-- **IMPORTANT:** Lua will only try __eq when the values being compared are *both tables.*
-- The result of the call is always converted to a boolean, so we can't return
-- a tensor then have .all() and .any() like numpy.
meta.__eq = function(a,b)
  local na, nb = number(a), number(b)
  if na ~= nil or nb ~= nil then
    return na == nb
  end

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

return Tensor
