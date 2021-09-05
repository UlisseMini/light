local light = {}

if not table.unpack then table.unpack = unpack end
if not table.pack then table.pack = function(...) return {...} end end

light.utils = require('light.utils')
light.numeric = require('light.numeric')
light.Tensor = require('light.tensor')
light.Value = require('light.value')

function light.no_grad(fn)
  local old = light.Value.grad_enabled
  light.Value.grad_enabled = false
  return light.utils.finally(fn, function()
    light.Value.grad_enabled = old
  end)
end

return light
