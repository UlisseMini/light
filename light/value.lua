local Value = {}
Value.__index = Value

function Value.__eq(a, b)
  return a.data == b.data
end

function Value:__tostring()
  return ('Value(%s, grad=%s)'):format(self.data, self.grad)
end

function Value:__len()
  error('cannot get length of ' .. tostring(self))
end

local ops = {}

local function op(name, forward, backward)
  Value['__' .. name] = function(a,b)
    a, b = Value(a), Value(b)
    return Value(forward(a.data, b.data), {_parents = {a, b}, _backward = backward})
  end
end

-- only differentiable ops are supported
op('add', function(a,b) return a+b  end, function(a,b) return 1, 1 end)
op('sub', function(a,b) return a-b  end, function(a,b) return 1, -1 end)
op('mul', function(a,b) return a*b  end, function(a,b) return b, a end)
op('div', function(a,b) return a/b  end, function(a,b) return 1/b, -a/b^2 end)
op('pow', function(a,b) return a^b  end, function(a,b) return b*a^(b-1), math.log(a)*a^b end)

function Value.new(data, args)
  if getmetatable(data) == Value then
    -- already a value, I don't think there's a usecase for nested values,
    -- and this simplifies forward (where I have to convert to value if not already value)
    return data
  end
  local self = {}
  self.data = data
  self.grad = nil

  args = args or {}
  self._parents = args._parents or {}
  self._backward = args._backward

  setmetatable(self, Value)
  return self
end
setmetatable(Value, {__call = function(self, ...) return self.new(...) end})

function Value:zero_grad()
  self.grad = Value(0)
  for _, parent in ipairs(self._parents) do
    parent:zero_grad()
  end
end

function Value:backward_no_zero()
  self.grad = Value(1)

  local stack = {self}
  while #stack > 0 do
    local node = table.remove(stack, 1)
    if node._backward then -- not a leaf node
      local parents = node._parents
      local derivs = table.pack(node._backward(table.unpack(parents)))
      if #derivs ~= #parents then
        error(('got %s derivs but have %s parents'):format(#derivs, #parents))
      end
      for i=1,#derivs do
        parents[i].grad = parents[i].grad + derivs[i] * node.grad

        table.insert(stack, parents[i])
      end
    end
  end
end

function Value:backward()
  self:zero_grad()
  self:backward_no_zero()
end

return Value

