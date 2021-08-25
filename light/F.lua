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

F.log = F:new(math.log, function(a, b) return 1/(a*F.log(b)), -F.log(a)/(b*F.log(b)^2) end)
F.add = F:new(function(a,b) return a+b  end, function(a,b) return 1, 1 end)
F.mul = F:new(function(a,b) return a*b  end, function(a,b) return b, a end)
F.pow = F:new(function(a,b) return a^b  end, function(a,b) return b*a^(b-1), F.log(a)*a^b end)
F.div = F:new(function(a,b) return a/b  end, function(a,b) return 1/b, -a/b^2 end)

F.relu = F:new(
  function(a) return math.max(0, a) end,
  function(a) if a > 0 then return 1 else return 0 end end
)

return F
