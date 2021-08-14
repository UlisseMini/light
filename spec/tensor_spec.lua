local light = dofile('light/init.lua')

describe('Tensor', function()
  describe('new', function()
    it('rejects tensors with non numbers', function()
      assert.error(function() light.Tensor({true}) end)
      assert.error(function() light.Tensor({{1, 'bar'}}) end)
    end)

    it('disallows nested tensors', function()
      -- TODO: maybe this should be allowed? It might screw up autodiff though
      local T = light.Tensor
      assert.error(function() T({T({1,2}), T({3,4})}) end)
    end)

    -- TODO
    -- it('raises an error on a badly shaped tensor', function()
    --   assert.error(function()
    --     local x = light.Tensor({{1,2}, {3,4,5}})
    --   end)
    -- end)

    it('does nothing to tensors', function()
      local t = light.Tensor({3,2})
      local t2 = light.Tensor(t)
      -- reference equality
      assert.is_true(rawequal(t, t2))
    end)
  end)

  describe('all', function()
    it('creates tensors from a vector shape', function()
      local t = light.Tensor.all({3}, 2)
      assert.equal({2,2,2}, t)
    end)

    it('creates tensors from a matrix shape', function()
      local t = light.Tensor.all({2,3}, 1)
      assert.equal({{1,1,1}, {1,1,1}}, t)
    end)
  end)

  describe('size', function()
    it('describes an array as a 1-tensor', function()
      local x = light.Tensor({1,2,3})
      assert.equal(x:size(), {3})
    end)

    it('describes a matrix as a 2-tensor', function()
      local x = light.Tensor({{1,2,3}, {3,4,5}})
      assert.equal(x:size(), {2, 3})
    end)
  end)

  --[[ TODO
  describe('indexing', function()
    local t
    before_each(function()
      -- we're mutating t here, to avoid chains of failure we reset t
      t = light.Tensor({1,2})
    end)

    it('raises an error when an index is out of bounds', function()
      assert.error(function() local x = t[3] end)
    end)

    it('raises an error when setting an index out of bounds', function()
      assert.error(function() t[3] = 4 end)
    end)

    it('allows normal mutation', function()
      t[1] = 2
      assert.is_equal(t, light.Tensor({2,2}))
    end)
  end)
  --]]

  describe('equality', function()
    local x = light.Tensor({3,2,1})

    it('should declare identical objects equal', function()
      assert.is_equal(x, x)
    end)

    it('should declare equal objects equal', function()
      local x2 = light.Tensor({3,2,1})
      assert.is_equal(x, x2)
    end)

    it('should declare not equal objects not equal', function()
      local x2 = light.Tensor({3,5,1})
      assert.are_not.equal(x, x2)
    end)

    it('should declare tensors with different dimensions not equal', function()
      local y = light.Tensor({3,2,1,1})
      assert.are_not.equal(x, y)
      assert.are_not.equal(y, x)
    end)

    it('should declare objects of different types not equal', function()
      assert.are_not.equal(x, 3)
      assert.are_not.equal(x, "foo")
    end)
  end)

  describe('fp', function()
    local t = light.Tensor({4,3,2})
    local t2 = light.Tensor({{1,2}, {3,4}})

    it('should reduce', function()
      assert.is_equal(t[1]*t[2]*t[3], t:reduce(function(acc, curr) return acc*curr end, 1))
    end)

    describe('map', function()
      it('should map over 1-tensors', function()
        local res = t:map(function(x) return x+1 end)
        assert.is_equal(res, t+1)
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
  end)

  describe('sum', function()
    it('should sum vectors', function()
      local t = light.Tensor({4,3,2})
      assert.is_equal(t:sum(), 9)
    end)

    -- TODO
    -- it('should sum matrices', function()
    --   local t = light.Tensor({{1,1}, {1,1}})
    --   assert.is_equal(t:sum(), 4)
    -- end)
  end)

  describe('dot', function()
    it('should take the dot product of two vectors', function()
      local t1 = light.Tensor({3,2,1})
      local t2 = light.Tensor({1,1,2})
      assert.is_equal(t1:dot(t2), 7)
    end)

    it('should raise an error for vectors of different sizes', function()
      local t1 = light.Tensor({3,2,1})
      local t2 = light.Tensor({1,1,2,4})
      assert.error(function() t1:dot(t2) end)
      assert.error(function() t2:dot(t1) end)
    end)
  end)

  describe('matmul', function()
    local m_2x2 = light.Tensor({{1,2}, {3,4}})
    local m_3x3 = light.Tensor({{1,2,3}, {3,4,5}, {5,3,2}})
    local m_3x2 = light.Tensor({{1,2}, {3,4}, {5,6}})
    local m_2x3 = light.Tensor({{1,2,3}, {4,5,6}})

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
  end)

  describe('Autodiff', function()
    it('should backprop through multiplication', function()
      local a = light.Tensor({1,2})
      local b = light.Tensor({2,3})
      local z = a * b
      z:backward()

      assert.equal(b, a.grad)
      assert.equal(a, b.grad)
    end)
  end)
end)

