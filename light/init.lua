local light = {}

if not table.unpack then table.unpack = unpack end
if not table.pack then table.pack = function(...) return {...} end end

light.utils = require('light.utils')
light.numeric = require('light.numeric')
light.Tensor = require('light.tensor')
light.Value = require('light.value')

return light
