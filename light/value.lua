local F = require('light.F')
local Value = {}
Value.__index = Value

local function set(xs)
  local t = {}
  for _, v in ipairs(xs) do t[v] = true end
  return t
end
local infix = set{'add', 'mul', 'div', 'sub', 'pow'}

-- TODO: fix F.log(Value) and math.log(Value) using multiple dispatch

-- Return x as a pure lua number if possible
function Value.item(x)
  if type(x) == 'number' then
    return x
  elseif Value.isinstance(x) then
    return x.data
  else
    error(('Do not know how to turn %s into a number'):format(x))
  end
end

local function allnums(t)
  local purely_numeric_args = {}
  for _, arg  in ipairs(t) do
    table.insert(purely_numeric_args, Value.item(arg))
  end

  return purely_numeric_args
end

for name, f in pairs(F) do
  if infix[name] then name = '__' .. name end

  Value[name] = function(...)
    local res = f(table.unpack(allnums({...})))
    return Value(res, {_parents = {...}, _backward = function(...)
      return f.derivs(...)
    end})
  end
end

function Value.__unm(a)   return -1 * a end
function Value.__eq(a, b) return Value.item(a) == Value.item(b) end
function Value.__lt(a, b) return Value.item(a) <  Value.item(b) end
function Value.__le(a, b) return Value.item(a) <= Value.item(b) end

function Value:__tostring()
  return ('Value(%s, grad=%s)'):format(self.data, self.grad)
end

function Value:__len()
  error('cannot get length of ' .. tostring(self))
end

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

  -- validate
  for i, p in ipairs(self._parents) do
    if getmetatable(p) ~= Value and type(p) ~= 'number' then
      error(('got %s in parents[%s] want Value or number'):format(p, i))
    end
  end

  setmetatable(self, Value)
  return self
end
setmetatable(Value, {__call = function(self, ...) return self.new(...) end})

function Value.isinstance(x)
  return getmetatable(x) == Value
end

function Value:zero_grad()
  self.grad = Value(0)
  for _, parent in ipairs(self._parents) do
    if Value.isinstance(parent) then
      parent:zero_grad()
    end
  end
end

function Value:backward_no_zero()
  self.grad = Value(1)

  local stack = {self}
  local visited = {}
  while #stack > 0 do
    local node = table.remove(stack, 1)
    -- TODO: Move indent lower down
    if not visited[node] and node._backward then -- not a leaf node, not visited
      local parents = node._parents
      local derivs = table.pack(node._backward(table.unpack(allnums(parents))))
      if #derivs ~= #parents then
        error(('got %s derivs but have %s parents'):format(#derivs, #parents))
      end

      for i=1,#derivs do
        if Value.isinstance(parents[i]) then
          -- only update grad if we have a value, otherwise it is a const
          -- and we don't need to update or add to the stack.
          parents[i].grad = parents[i].grad + derivs[i] * node.grad

          table.insert(stack, parents[i])
        end
      end
    end
    visited[node] = true
  end
end

function Value:backward()
  self:zero_grad()
  self:backward_no_zero()
end

return Value

