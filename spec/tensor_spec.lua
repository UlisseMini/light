local light = require('light')
local T = light.Tensor


local m_2x2 = light.Tensor({{1,2}, {3,4}})
local m_3x3 = light.Tensor({{1,2,3}, {3,4,5}, {5,3,2}})
local m_3x2 = light.Tensor({{1,2}, {3,4}, {5,6}})
local m_2x3 = light.Tensor({{1,2,3}, {4,5,6}})


describe('Tensor', function()
  describe('new', function()
    it('rejects tensors with non numbers', function()
      assert.error(function() light.Tensor({true}) end)
      assert.error(function() light.Tensor({{1, 'bar'}}) end)
    end)

    it('disallows nested tensors', function()
      assert.error(function() T({T({1,2}), T({3,4})}) end)
    end)

    it('allows scalar tensors', function()
      assert.not_error(function() light.Tensor(5) end)
    end)

    -- TODO
    -- it('raises an error on a badly shaped tensor', function()
    --   assert.error(function()
    --     local x = light.Tensor({{1,2}, {3,4,5}})
    --   end)
    -- end)
  end)

  describe('all', function()
    it('creates tensors from a vector shape', function()
      local t = light.Tensor.all({3}, 2)
      assert.equal(T{2,2,2}, t)
    end)

    it('creates tensors from a matrix shape', function()
      local t = light.Tensor.all({2,3}, 1)
      assert.equal(T{{1,1,1}, {1,1,1}}, t)
    end)
  end)

  describe('size', function()
    describe('gives a scalar tensor shape {}', function()
      local x = T(1)
      assert.are.same({}, x:size())
    end)

    it('describes an array as a 1-tensor', function()
      local x = light.Tensor({1,2,3})
      assert.are.same({3}, x:size())
    end)

    it('describes a matrix as a 2-tensor', function()
      local x = light.Tensor({{1,2,3}, {3,4,5}})
      assert.are.same({2, 3}, x:size())
    end)
  end)

  describe('indexing', function()
    local t
    before_each(function()
      -- we're mutating t here, to avoid chains of failure we reset t
      t = light.Tensor({1,2})
    end)

    it('reads numbers', function()
      assert.equal(1, t[1])
      assert.equal(2, t[2])

      assert.equal(1, m_2x2[1][1])
      assert.equal(2, m_2x2[1][2])
      assert.equal(3, m_2x2[2][1])
      assert.equal(4, m_2x2[2][2])
    end)

    it('returns nil when an index is out of bounds', function()
      assert.equal(nil, t[5])
    end)
    
    it('raises an error when setting an index out of bounds', function()
      assert.error(function() t[3] = 4 end)
    end)

    it('allows index mutation', function()
      t[1] = 2
      assert.is_equal(t, light.Tensor({2,2}))
    end)
  end)

  describe('equality', function()
    local x = light.Tensor({3,2,1})

    it('should handle misshapen tensors', function()
      assert.is_false(T{} == T{1})
      assert.is_false(T{} == T(1))
      assert.is_false(T{1,2} == T(1))
      assert.is_false(T{{}} == T{1})
    end)

    it('should declare identical objects equal', function()
      assert.is_true(x == x)
    end)

    it('should declare equal objects equal', function()
      local x2 = light.Tensor({3,2,1})
      assert.is_true(x == x2)
    end)

    it('should declare not equal objects not equal', function()
      local x2 = light.Tensor({3,5,1})
      assert.is_true(x ~= x2)
    end)

    it('should declare tensors with different dimensions not equal', function()
      local y = light.Tensor({3,2,1,1})
      assert.is_true(x ~= y)
      assert.is_true(y ~= x)
    end)

    it('should declare objects of different types not equal', function()
      assert.are_not.equal(x, 3)
      assert.are_not.equal(x, "foo")
    end)

    it('compares equality of scalar tensors', function()
      assert.equal(T(5), T(5))
      assert.not_equal(T(5), T(6))
    end)
  end)

  describe('fp', function()
    local t = light.Tensor({4,3,2})
    local t2 = light.Tensor({{1,2}, {3,4}})

    describe('map', function()
      it('should map over 1-tensors', function()
        local res = t:map(function(x) return x+1 end)
        assert.is_equal(res, t+1)

        local res = t:map(function(x) return x*2 end)
        assert.equal(res, T{8,6,4})
      end)
    end)
  end)

  describe('ops', function()
    local t = light.Tensor({1,2,3})
    it('should broadcast numbers over tensors', function()
      assert.equal(t + 1, light.Tensor({2,3,4}))
      assert.equal(t / 2, light.Tensor({1/2,2/2,3/2}))
    end)

    it('should apply operations piecewise', function()
      assert.equal(t + t, t*2)
      assert.equal(t / light.Tensor({2,2,1}), light.Tensor({1/2,2/2,3/1}))
    end)

    it('should apply operations to 0-tensors', function()
      local t = T(4) / 2
      assert.equal(T(2), t)
      assert.equal(t, T(2))
    end)
  end)

  describe('sum', function()
    it('should sum vectors', function()
      local t = light.Tensor({4,3,2})
      assert.is_true(T(9) == t:sum())
    end)

    -- TODO
    -- it('should sum matrices', function()
    --   local t = light.Tensor({{1,1}, {1,1}})
    --   assert.is_equal(t:sum(), 4)
    -- end)
  end)

  describe('prod', function()
    it('should take the product of vectors', function()
      local t = T {1,2,3}
      assert.equal(T(6), t:prod())
    end)
  end)

  describe('dot', function()
    it('should take the dot product of two vectors', function()
      local t1 = light.Tensor({3,2,1})
      local t2 = light.Tensor({1,1,2})
      assert.is_equal(t1:dot(t2):item(), 7)
    end)

    it('should raise an error for vectors of different sizes', function()
      local t1 = light.Tensor({3,2,1})
      local t2 = light.Tensor({1,1,2,4})
      assert.error(function() t1:dot(t2) end)
      assert.error(function() t2:dot(t1) end)
    end)
  end)

  describe('matmul', function()
    it('should raise an error when the matrix sizes do not match', function()
      assert.error(function() m_2x2:matmul(m_3x3) end)
      assert.error(function() m_3x3:matmul(m_2x2) end)

      assert.error(function() m_3x2:matmul(m_3x3) end)
    end)

    it('should multiply 2x2 matrices', function()
      local got = m_2x2:matmul(m_2x2)
      local want = light.Tensor({{7, 10}, {15, 22}})
      assert.is_equal(want, got)
    end)

    it('should multiply (2x3) * (3x2)', function()
      local got = m_2x3:matmul(m_3x2)
      local want = light.Tensor({{22, 28}, {49, 64}})
      assert.is_equal(want, got)
    end)

    -- it('should do matrix vector multiplication', function()
    --   local v = light.Tensor({1,1})
    --   local got = m_2x2:matmul(v)
    --   local want = light.Tensor({3, 7})
    --   assert.is_equal(want, got)
    -- end)

    it('should do (2 by 2) matmul (2 by 1)', function()
      local v = T({{1}, {1}})
      local got = m_2x2:matmul(v)
      local want = T({{3}, {7}})
      assert.is_equal(want, got)
    end)

    it('should do outer products (2 by 1) * (1 by 2) = (2 by 2)', function()
      local v = T {
        {1},
        {2},
      }
      local w = T {
        {3, 4},
      }
      local want = T {
        {3, 4},
        {6, 8},
      }
      local got = v:matmul(w)

      assert.equal(want, got)
    end)
  end)

  describe('Autodiff', function()
    after_each(function() light.Tensor.do_grad = true end)

    it('should backprop through multiplication', function()
      local a = light.Tensor({1,2})
      local b = light.Tensor({2,3})
      local z = a * b
      z:backward()

      assert.equal(b, a.grad)
      assert.equal(a, b.grad)
    end)

    it('should backprop through addition', function()
      local a = light.Tensor({1,2})
      local b = light.Tensor({2,3})
      local z = a + b
      z:backward()

      local ones = light.Tensor.ones({2})
      assert.equal(ones, a.grad)
      assert.equal(ones, b.grad)
    end)

    it('should backprop through multiplication and addition', function()
      local x,y,a,b,z

      x = light.Tensor({1,2})
      y = light.Tensor({2,3})

      a = x * y
      b = x + y

      z = a * b
      z:backward()

      -- this is what autodiff should have done, spelled out. where each
      -- d(var) is dz/d(var).
      light.Tensor.do_grad = false
      local dx,dy,da,db,dz
      dz = light.Tensor.ones({2})
      db = a
      da = b
      dx = y*da + db
      dy = x*da + db

      assert.equal(dx, x.grad)
      assert.equal(dy, y.grad)
    end)

    it('should have correct numerical derivatives for piecewise ops', function()
      math.randomseed(42)

      local N = 42 -- how many random tests to preform on each op

      local ops = {'add', 'sub', 'mul', 'div', 'pow'}
      local meta = getmetatable(light.Tensor({1}))

      for _, name in ipairs(ops) do
        local op = meta['__' .. name]

        local a, b
        local partial_a = light.numeric.derivative(function(x) return op.op(x, b) end)
        local partial_b = light.numeric.derivative(function(x) return op.op(a, x) end)
        for i=1,N do
          -- we shift by 1 so that division doesn't blow up
          a, b = math.random(), math.random() + 1

          local da, db = op.derivs(a, b)
          local numeric_da, numeric_db = partial_a(a), partial_b(b)

          local error_a, error_b = math.abs(numeric_da - da), math.abs(numeric_db - db)

          assert.is_true(error_a <= 0.0001, ('%s(%s, %s): error_a: %s'):format(name, a, b, error_a))
          assert.is_true(error_b <= 0.0001, ('%s(%s, %s): error_b: %s'):format(name, a, b, error_b))
        end
      end
    end)

    it('should backprop through dot products', function()
      local x = light.Tensor({1,2})
      local y = light.Tensor({2,3})
      local z = x:dot(y)
      z:backward()
      assert.is_equal(x, y.grad)
      assert.is_equal(y, x.grad)
    end)

    it('should backprop through sums', function()
      local x = light.Tensor({1,2})
      local z = x:sum()
      z:backward()
      assert.equal(T{1, 1}, x.grad)
    end)

    -- TODO
    -- it('should backprop through products', function()
    --   local x = light.Tensor({1,2})
    --   local z = x:prod()
    --   z:backward()
    --   assert.equal(x.grad, T({2, 1}))
    -- end)

    it('should backprop through dot products and arithmetic', function()
      local x = light.Tensor({1,2})
      local y = light.Tensor({2,3})

      local a = x * y
      local b = x + y

      local q = a:dot(b)
      local z = 2*q

      z:backward()

      local dz = T(1)
      local dq = 2 * dz
      local db = a * dq
      local da = b * dq
      local dy = x * da + db
      local dx = y * da + db

      assert.equal(dq, q.grad)
      assert.equal(db, b.grad)
      assert.equal(da, a.grad)
      assert.equal(dy, y.grad)
      assert.equal(dx, x.grad)
    end)

    -- it('should backprop through matrix multiplication', function()
    --   local A = T({{1,2}, {3,4}})
    --   local B = T({{1,7}, {2,5}})
    --   local x = T({1,1})

    --   local C = A:matmul(B)
    --   local y = C:matmul(x)
    --   local z = y:sum()

    --   z:backward()
    --   assert.equal(T({1, 1}), y.grad)
    --   assert.equal(T({}), C.grad)
    -- end)
  end)
end)

