local V = require('light.value')

-- todo: refactor to numerics? register assert.close to luaassert
local function assert_close(want, got)
  want, got = V.tonumber(want), V.tonumber(got)
  local tol = 0.01
  local err = want - got
  if math.abs(err) > tol then
    error(('got %s want %s diff %s'):format(got, want, err))
  end
end

describe('Value', function()
  it('has equality', function()
    assert.equal(V(5), V(5))
    assert.not_equal(V(5), V(4))
    -- no comparing different types
    assert.error(function() return V(5) == {} end)
    assert.error(function() return V(5) == {data = 5} end)
  end)

  it('has inequality', function()
    assert.is_true(V(1) <= V(2))
    assert.is_true(V(1) < V(2))
    assert.is_false(V(2) < V(2))
    assert.is_true(V(2) <= V(2))
    assert.is_true(V(2) >= V(2))
    assert.is_false(V(2) > V(2))
    assert.is_true(V(3) > V(2))
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

  it('backprops through relu', function()
    local x = V(4)
    local z = x:relu()
    z:backward()
    assert.equal(1, x.grad.data)

    x.data = -3
    z:backward()
    assert.equal(0, x.grad.data)
  end)

  it('backprops through math library functions', function()
    local x,y,z
    x = V(3)
    y = x:log()
    z = y:atan()
    z:backward()

    local want = (1/x)/(1 + y^2)
    assert_close(want, x.grad)
  end)

  it('backprops through exp and powers', function()
    local x = V(2)
    local y = V(3)
    local z = x^y
    z:backward()

    assert_close(12, x.grad)
    assert_close(math.log(x.data)*z, y.grad)
  end)

  it('passes the micrograd example', function()
    local a,b,c,d,e,f,g

    a = V(-4.0)
    b = V(2.0)
    c = a + b
    d = a * b + b^3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a):relu()
    d = d + 3 * d + (b - a):relu()
    e = c - d
    f = e^2
    g = f / 2.0
    g = g + 10.0 / f

    assert_close(24.7041, g.data)
    g:backward()
    assert_close(138.8338, a.grad.data)
    assert_close(645.5773, b.grad.data)
  end)

  describe('graphs using graphviz', function()
    it('should not crash on wierd inputs', function()
      local _ = V(5):graphviz_dot()
      local _ = (V(5) + V(3)):graphviz_dot()
    end)

    local function run_dot(dot)
      local file = assert(io.popen('dot &> /tmp/light.busted.out', 'w'))
      file:write(dot)
      file:flush()
      local ok, msg, code = file:close()
      local file = assert(io.open('/tmp/light.busted.out', 'r'))
      local output = file:read()

      return ok, ('%s (%s %s)'):format(output, msg, code)
    end

    it('should output valid dot language code', function()
      local a = V(3)
      local b = V(5)
      local c = V(-0.3)
      local z = a + b * c
      z = z + b * c

      assert(run_dot(z:graphviz_dot()))
      z:backward()
      assert(run_dot(z:graphviz_dot()))
    end)
  end)
end)
