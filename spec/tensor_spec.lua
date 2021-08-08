local torch = require('./torch')

describe('Tensor', function()
  describe('new', function()
    it('rejects tensors with non numbers', function()
      assert.error(function() torch.Tensor({true}) end)
      assert.error(function() torch.Tensor({{1, 'bar'}}) end)
    end)

    -- TODO
    -- it('raises an error on a badly shaped tensor', function()
    --   assert.error(function()
    --     local x = torch.Tensor({{1,2}, {3,4,5}})
    --   end)
    -- end)
  end)

  describe('size', function()
    it('describes an array as a 1-tensor', function()
      local x = torch.Tensor({1,2,3})
      assert.are.same(x:size(), {3})
    end)

    it('describes a matrix as a 2-tensor', function()
      local x = torch.Tensor({{1,2,3}, {3,4,5}})
      assert.are.same(x:size(), {2, 3})
    end)
  end)

  --[[ TODO
  describe('indexing', function()
    local t
    before_each(function()
      -- we're mutating t here, to avoid chains of failure we reset t
      t = torch.Tensor({1,2})
    end)

    it('raises an error when an index is out of bounds', function()
      assert.error(function() local x = t[3] end)
    end)

    it('raises an error when setting an index out of bounds', function()
      assert.error(function() t[3] = 4 end)
    end)

    it('allows normal mutation', function()
      t[1] = 2
      assert.is_equal(t, torch.Tensor({2,2}))
    end)
  end)
  --]]

  describe('equality', function()
    local x = torch.Tensor({3,2,1})

    it('should declare identical objects equal', function()
      assert.is_equal(x, x)
    end)

    it('should declare equal objects equal', function()
      local x2 = torch.Tensor({3,2,1})
      assert.is_equal(x, x2)
    end)

    it('should declare not equal objects not equal', function()
      local x2 = torch.Tensor({3,5,1})
      assert.are_not.equal(x, x2)
    end)

    it('should declare tensors with different dimensions not equal', function()
      local y = torch.Tensor({3,2,1,1})
      assert.are_not.equal(x, y)
      assert.are_not.equal(y, x)
    end)

    it('should declare objects of different types not equal', function()
      assert.are_not.equal(x, 3)
      assert.are_not.equal(x, "foo")
    end)
  end)

  describe('fp', function()
    local t = torch.Tensor({4,3,2})
    local t2 = torch.Tensor({{1,2}, {3,4}})

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
    local t = torch.Tensor({1,2,3})
    it('should broadcast numbers over tensors', function()
      assert.is_equal(t + 1, torch.Tensor({2,3,4}))
      assert.is_equal(t / 2, torch.Tensor({1/2,2/2,3/2}))
    end)

    it('should apply operations piecewise', function()
      assert.is_equal(t + t, t*2)
      assert.is_equal(t / torch.Tensor({2,2,1}), torch.Tensor({1/2,2/2,3/1}))
    end)
  end)
end)

