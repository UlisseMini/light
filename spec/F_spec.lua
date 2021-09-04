local F = require('light.F')
local numeric = require('light.numeric')
local utils = require('light.utils')

-- Test a function f : R^n -> R
local function test_fn_deriv(f, nargs, low, high)
  local low, high = low or -3, high or 3
  assert(low < high)
  math.randomseed(42)

  -- todo: change numeric.gradient to pass f params like this?
  local wrapf = function(xs) return f(table.unpack(xs)) end

  for i=1, 100 do
    local args = {}
    for j=1,nargs do
      args[j] = math.random()*(high-low) + low
    end

    local derivs = {f.derivs(table.unpack(args))}
    assert(#derivs == nargs)
    local numeric_derivs = numeric.gradient(wrapf)(args)
    assert(#numeric_derivs == nargs)

    for j=1,nargs do
      local d_analytic = derivs[j]
      local d_numeric = numeric_derivs[j]

      -- print(math.abs(d_analytic - d_numeric) < 0.001)
      local distance = math.abs(d_numeric - d_analytic)
      local close = distance < 0.01
      if not close then
        error(('[i:%s,j:%s] want %s got %s at %s'):format(i, j, d_numeric, d_analytic, utils.pp(args)))
      end
    end
  end
end

describe('F', function()
  it('should have correct numerical derivatives', function()
    test_fn_deriv(F.add, 2)
    test_fn_deriv(F.sub, 2)
    test_fn_deriv(F.mul, 2)
    test_fn_deriv(F.div, 2, 0.1, 3)

    -- math library
    test_fn_deriv(F.log, 2, 1, 10)
    test_fn_deriv(F.log10, 1, 1, 100)
    test_fn_deriv(F.pow, 2, 0, 3)

    test_fn_deriv(F.cosh, 1, -2, 2)
    test_fn_deriv(F.sinh, 1, -2, 2)
    test_fn_deriv(F.tanh, 1, -2, 2)

    test_fn_deriv(F.acos, 1, -1, 1)
    test_fn_deriv(F.asin, 1, -1, 1)
    test_fn_deriv(F.atan, 1, -5, 5)

    test_fn_deriv(F.rad, 1, -1000, 1000)
    test_fn_deriv(F.deg, 1, -math.pi*4, math.pi*4)

    test_fn_deriv(F.exp, 1, -math.pi*2, math.pi*2)
    test_fn_deriv(F.cos, 1, -math.pi*2, math.pi*2)
    test_fn_deriv(F.sin, 1, -math.pi*2, math.pi*2)
    test_fn_deriv(F.tan, 1, -math.pi*2, math.pi*2)

    test_fn_deriv(F.abs, 1, -5, 5)
    test_fn_deriv(F.max, 4, -5, 5)
    test_fn_deriv(F.min, 4, -5, 5)

    test_fn_deriv(F.sqrt, 1, 0, 10)

    -- misc
    test_fn_deriv(F.relu, 1)
  end)
end)
