local F = require('light.F')
local utils = require('light.utils')
local Value = {
  grad_enabled = true, -- contextually limit autograd
}
Value.__index = Value

local function isnan(x) return x ~= x end
local function isfinite(x) return x > -math.huge and x < math.huge end
local function set(xs)
  local t = {}
  for _, v in ipairs(xs) do t[v] = true end
  return t
end
local infix = set{'add', 'mul', 'div', 'sub', 'pow'}

-- Return x as a pure lua number if possible
function Value.tonumber(x)
  if type(x) == 'number' then
    return x
  elseif getmetatable(x) == Value then
    return x.data
  else
    error(('Do not know how to turn %s into a number'):format(x))
  end
end

local function allnums(t)
  local purely_numeric_args = {}
  for _, arg  in ipairs(t) do
    table.insert(purely_numeric_args, Value.tonumber(arg))
  end

  return purely_numeric_args
end

for name, f in pairs(F) do
  if infix[name] then name = '__' .. name end

  Value[name] = function(...)
    local res = f(table.unpack(allnums({...})))
    return Value(res, {_parents = {...}, _backward = f.derivs, _op = name})
  end
end

function Value.__unm(a)   return -1 * a end
function Value.__eq(a, b) return Value.tonumber(a) == Value.tonumber(b) end
function Value.__lt(a, b) return Value.tonumber(a) <  Value.tonumber(b) end
function Value.__le(a, b) return Value.tonumber(a) <= Value.tonumber(b) end

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
  if isnan(data) or not isfinite(data) then
    error(('attempt to create Value(%s) args=%s'):format(data, utils.pp(args)))
  end
  self.data = data
  self.grad = nil

  args = args or {}
  -- WARNING: don't do requires_grad = args.requires_grad or true since args.requires_grad=false
  -- counts as falsey, so you'll always be setting it to true!
  if args.requires_grad ~= nil then
    self.requires_grad = args.requires_grad
  else
    self.requires_grad = true
  end

  if self.requires_grad and Value.grad_enabled then
    self._parents = args._parents
    self._backward = args._backward
    self._op = args._op -- the op that produced this node, for debugging/etc
  end
  self._parents = self._parents or {}

  -- Replace numbers in _parents with Value(requires_grad=false)
  for i=1,#self._parents do
    if type(self._parents[i]) == 'number' then
      self._parents[i] = Value(self._parents[i], {requires_grad=false})
    end
  end


  setmetatable(self, Value)
  return self
end
setmetatable(Value, {__call = function(self, ...) return self.new(...) end})

Value.get = {}
function Value.get.grad(x)
  if getmetatable(x) == Value then
    return x.grad.data
  else
    return nil
  end
end
function Value.get.data(x)
  if type(x) == 'number' then
    return x
  elseif getmetatable(x) == Value then
    return x.data
  else
    return nil
  end
end


function Value:zero_grad()
  self.grad = Value(0)
  local visited = {}
  for _, parent in ipairs(self._parents) do
    if not visited[parent] and parent.requires_grad then
      parent:zero_grad()
      visited[parent] = true
    end
  end
end

-- top down topological ordering of the dag, (nodes with no parents, children of those nodes, ...)
-- TODO: make into an iterator
function Value:topo()
  local topo = {}
  local visited = {}

  local function build_topo(node)
    if not visited[node] then
      visited[node] = true
      for _, parent in ipairs(node._parents) do
        build_topo(parent)
      end
      table.insert(topo, node)
    end
  end
  build_topo(self)

  return topo
end

function Value:backward_no_zero()
  assert(Value.grad_enabled, 'attempt to backprop when grad is disabled')

  self.grad = Value(1)

  -- walk in reverse topological order, ie. always call backward on all children
  -- before calling backward on their parents
  local topo = self:topo()
  for i=#topo,1,-1 do
    local node = topo[i]
    if node._backward then
      local parents = node._parents

      -- TODO: Only compute derivatives we need
      local derivs = table.pack(node._backward(table.unpack(allnums(parents))))
      if #derivs < #parents then
        error(('got %s derivs but have %s parents'):format(#derivs, #parents))
      end

      for i=1,#parents do
        if parents[i].requires_grad then
          -- my sanity is worth a few cpu cycles
          if isnan(derivs[i]) or not isfinite(derivs[i]) then
            print(node:_debug())
            error(('derivs[%s] = %s, backprop through %s'):format(i, derivs[i], node._op))
          end
          parents[i].grad = parents[i].grad + derivs[i] * node.grad
        end
      end
    end
  end
end

function Value:backward()
  self:zero_grad()
  self:backward_no_zero()
end

function Value:_debug()
  return ('%s = %s(%s, %s)')
    :format(self.data, self._op, (self._parents[1] or {}).data, (self._parents[2] or {}).data)
end

function Value.pretty_op(op)
  if op:sub(1, 2) == '__' then op = op:sub(3, -1) end
  return op
end

function Value._id(t)
  local m = getmetatable(t)
  setmetatable(t, nil)
  local n = tonumber(tostring(t):sub(8, -1))
  setmetatable(t, m)
  return n
end

function Value:graphviz_dot()
  local s = 'digraph {\n'
  s = s .. 'rankdir=LR\n'

  local function node_label(node)
    if node.requires_grad then
      local grad = (node.grad or {}).data
      return ('{data %.3f | grad %s}'):format(node.data, grad)
    else
      return ('const %.3f'):format(node.data)
    end
  end


  local topo = self:topo()
  for _, node in ipairs(topo) do
    -- this node
    s = s .. ('%s [shape=record,label="%s"]\n'):format(node:_id(), node_label(node))
    if node._op then
      local op = node._op
      local op_id = op .. node:_id()
      s = s .. ('"%s" [label="%s"]\n'):format(op_id, Value.pretty_op(op))
      s = s .. ('"%s" -> "%s"\n'):format(op_id, node:_id())

      for _, parent in ipairs(node._parents) do
        s = s .. ('"%s" -> "%s"\n'):format(parent:_id(), op_id)
      end
    end
  end

  s = s .. '}\n'
  return s
end

return Value

