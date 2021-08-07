local torch = require('./torch')

describe('tensor equality', function()
  local x = torch.Tensor:new({3,2,1})

  it('should declare identical objects equal', function()
    assert.is_equal(x, x)
  end)

  it('should declare equal objects equal', function()
    local x2 = torch.Tensor:new({3,2,1})
    assert.is_equal(x, x2)
  end)

  it('should declare not equal objects not equal', function()
    local x2 = torch.Tensor:new({3,5,1})
    assert.are_not.equal(x, x2)
  end)

  it('should declare tensors with different dimensions not equal', function()
    local y = torch.Tensor:new({3,2,1,1})
    assert.are_not.equal(x, y)
    assert.are_not.equal(y, x)
  end)

  it('should declare objects of different types not equal', function()
    assert.are_not.equal(x, 3)
    assert.are_not.equal(x, "foo")
  end)
end)

describe('tensor fp', function()
  local t = torch.Tensor:new({4,3,2})
  it('should reduce', function()
    assert.is_equal(t[1]*t[2]*t[3], t:reduce(function(acc, curr) return acc*curr end, 1))
  end)

  it('should map', function()
    local res = t:map(function(x) return x+1 end)
    assert.is_equal(res, t+1)
  end)
end)
