local V = require('light.value')

local a,b,c,d
a = V(-4.0)
b = V(2.0)
c = a + b
d = c * b + a:relu()

d:backward()
d:graphviz_dot()
