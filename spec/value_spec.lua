
local V = require('light.value')

describe('Value', function()
  it('has equality', function()
    assert.equal(V(5), V(5))
    assert.not_equal(V(5), V(4))
    assert.not_equal(V(5), {})

    -- not sure, but this is the current implementation
    assert.equal(V(5), {data = 5})
  end)

  it('does math', function()
    assert.equal(V(5) + 2, V(7))
    assert.equal(V(5) + V(2), V(7))
    assert.equal(V(4) / V(2), V(2))
    assert.equal(4 / V(2), V(2))
  end)

  it('backprops through multiplication', function()
    local a = V(3)
    local b = V(2)
    local z = a * b
    z:backward()

    assert.equal(b, a.grad)
    assert.equal(a, b.grad)
  end)

  it('backprops through multiplication and addition', function()
      local x,y,a,b,z

      x = V(3)
      y = V(2)

      a = x * y
      b = x + y

      z = a * b
      z:backward()

      -- this is what autodiff should have done, spelled out
      local dx,dy,da,db,dz
      dz = V(1)
      db = a
      da = b
      dx = y*da + db
      dy = x*da + db

      assert.equal(dx.data, x.grad.data)
      assert.equal(dy.data, y.grad.data)
  end)
end)
