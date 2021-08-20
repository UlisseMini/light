local numeric = require('light.init').numeric

local fns = {
  ['x^2'] = function(x) return x^2 end,
  ['x^3 + x^2'] = function(x) return x^3 + x^2 end,
  ['x^3 + y^2'] = function(t) return t[1]^3 + t[2]^2 end,
}

describe('numeric', function()
  describe('round', function()
    it('should round numbers up', function()
      assert.equal(4, numeric.round(3.9))
      assert.equal(3.9, numeric.round(3.89, 1))
    end)
    it('should round numbers down', function()
      assert.equal(3, numeric.round(3.4))
      assert.equal(3.4, numeric.round(3.44, 1))
      assert.equal(3.4, numeric.round(3.44444444444444, 1))
    end)

    -- the normal mathematical convention
    it('should round up when equal', function()
      assert.equal(4, numeric.round(3.5))
      assert.equal(3.6, numeric.round(3.55, 1))
    end)
  end)

  describe('derivative', function()
    it('should take the derivative of polynomials', function()
      local d1 = numeric.derivative(fns['x^2'])
      assert.equal(4, numeric.round(d1(2)))
      assert.equal(6, numeric.round(d1(3)))

      local d2 = numeric.derivative(fns['x^3 + x^2'])
      assert.equal(5, numeric.round(d2(1)))
    end)
  end)

  describe('gradient', function()
    it('should take the gradient of polynomials', function()
      local f = fns['x^3 + y^2'] -- true gradient: 3*x^2 + 2*y
      local nablaf = numeric.gradient(f)
      assert.are.same({12, 4}, numeric.round(nablaf({2, 2})))
      assert.are.same({3, 4}, numeric.round(nablaf({1, 2})))
    end)
  end)
end)
