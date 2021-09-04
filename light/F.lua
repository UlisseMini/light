-- differentiable functions R^n -> R (that expect numbers and give numbers)
local F = {}

function F:new(func, derivs)
  local f = {}
  assert(func ~= nil, 'func is nil')
  assert(derivs ~= nil, 'got nil derivs, F is only for differentiable functions')

  setmetatable(f, {
      __call = function(_, ...)
        return func(...)
      end
    })
  f.derivs = derivs
  return f
end

-- Basic infix ops
F.add = F:new(function(a,b) return a+b  end, function(a,b) return 1, 1 end)
F.sub = F:new(function(a,b) return a-b  end, function(a,b) return 1, -1 end)
F.mul = F:new(function(a,b) return a*b  end, function(a,b) return b, a end)
F.div = F:new(function(a,b) return a/b  end, function(a,b) return 1/b, -a/b^2 end)


-- math library functions

F.log = F:new(math.log, function(a, b)
  b = b or math.exp(1)
  return 1/(a*F.log(b)), -F.log(a)/(b*F.log(b)^2)
end)
F.rad = F:new(math.rad, function(x) return math.pi/180 end)
F.deg = F:new(math.deg, function(x) return 180/math.pi end)
F.exp = F:new(math.exp, math.exp)

F.sqrt = F:new(math.sqrt, function(x) return 0.5/math.sqrt(x) end)
F.pow = F:new(math.pow, function(a,b) return b*a^(b-1), F.log(a)*a^b end)
F.log10 = F:new(math.log10, function(x) return 1/(x*math.log(10)) end)

F.cos = F:new(math.cos, function(x) return -math.sin(x) end)
F.sin = F:new(math.sin, function(x) return math.cos(x) end)
F.tan = F:new(math.tan, function(x) return math.cos(x)^-2 end)

F.atan = F:new(math.atan, function(x) return 1/(1 + x^2) end)
F.acos = F:new(math.acos, function(x) return -1/math.sqrt(1 - x^2) end)
F.asin = F:new(math.asin, function(x) return 1/math.sqrt(1 - x^2) end)

F.cosh = F:new(math.cosh, math.sinh)
F.sinh = F:new(math.sinh, math.cosh)
F.tanh = F:new(math.tanh, function(x) return 1/math.cosh(x)^2 end)

F.abs = F:new(math.abs, function(x) if x > 0 then return 1 else return -1 end end)

-- TODO: shorten this code, the concept is simple, 1 for max 0 for others.
F.max = F:new(math.max, function(...)
  local m = math.max(...)
  local ret = {}
  for i,v in ipairs{...} do
    if v < m then ret[i] = 0 else ret[i] = 1 end
  end
  return table.unpack(ret)
end)
F.min = F:new(math.min, function(...)
  local m = math.min(...)
  local ret = {}
  for i,v in ipairs{...} do
    if v > m then ret[i] = 0 else ret[i] = 1 end
  end
  return table.unpack(ret)
end)

-- obscure but also differentiable: math.ldexp, math.frexp 

-- misc
F.relu = F:new(
  function(a) return math.max(0, a) end,
  function(a) if a > 0 then return 1 else return 0 end end
)

return F
