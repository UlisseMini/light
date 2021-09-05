-- differentiable functions R^n -> R (that expect numbers and give numbers)
local F = {}

function F:def(name, func, derivs)
  local f = {}
  assert(func ~= nil, 'func is nil')
  assert(derivs ~= nil, 'got nil derivs, F is only for differentiable functions')

  setmetatable(f, {
      __call = function(_, ...)
        return func(...)
      end
    })
  f.derivs = derivs
  F[name] = f
end

-- Basic infix ops
F:def('add', function(a,b) return a+b  end, function(a,b) return 1, 1 end)
F:def('sub', function(a,b) return a-b  end, function(a,b) return 1, -1 end)
F:def('mul', function(a,b) return a*b  end, function(a,b) return b, a end)
F:def('div', function(a,b) return a/b  end, function(a,b) return 1/b, -a/b^2 end)


-- math library functions

F:def('log', math.log, function(a, b)
  b = b or math.exp(1)
  return 1/(a*F.log(b)), -F.log(a)/(b*F.log(b)^2)
end)
F:def('rad', math.rad, function(x) return math.pi/180 end)
F:def('deg', math.deg, function(x) return 180/math.pi end)
F:def('exp', math.exp, math.exp)

F:def('sqrt', math.sqrt, function(x) return 0.5/math.sqrt(x) end)
F:def('pow', math.pow, function(a,b) return b*a^(b-1), F.log(a)*a^b end)
F:def('log10', math.log10, function(x) return 1/(x*math.log(10)) end)

F:def('cos', math.cos, function(x) return -math.sin(x) end)
F:def('sin', math.sin, function(x) return math.cos(x) end)
F:def('tan', math.tan, function(x) return math.cos(x)^-2 end)

F:def('atan', math.atan, function(x) return 1/(1 + x^2) end)
F:def('acos', math.acos, function(x) return -1/math.sqrt(1 - x^2) end)
F:def('asin', math.asin, function(x) return 1/math.sqrt(1 - x^2) end)

F:def('cosh', math.cosh, math.sinh)
F:def('sinh', math.sinh, math.cosh)
F:def('tanh', math.tanh, function(x) return 1/math.cosh(x)^2 end)

F:def('abs', math.abs, function(x) if x > 0 then return 1 else return -1 end end)

-- TODO: shorten this code, the concept is simple, 1 for max 0 for others.
F:def('max', math.max, function(...)
  local m = math.max(...)
  local ret = {}
  for i,v in ipairs{...} do
    if v < m then ret[i] = 0 else ret[i] = 1 end
  end
  return table.unpack(ret)
end)
F:def('min', math.min, function(...)
  local m = math.min(...)
  local ret = {}
  for i,v in ipairs{...} do
    if v > m then ret[i] = 0 else ret[i] = 1 end
  end
  return table.unpack(ret)
end)

-- obscure but also differentiable: math.ldexp, math.frexp

-- misc
F:def('relu',
  function(a) return math.max(0, a) end,
  function(a) if a > 0 then return 1 else return 0 end end
)

return F
