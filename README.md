# light

Machine learning in pure lua. Think pytorch but a lot slower

# Examples

## Least Squares

Take a look at `examples/least_squares.py` for the Pytorch version (which yields the exact same solution, within rounding error)

```
$ lua examples/least_squares.lua
loss 5.0000     x = {0, 0}
loss 0.8200     x = {0.14, 0.2}
loss 0.1422     x = {0.196, 0.2808}
loss 0.0322     x = {0.218176, 0.3136}
[...many more lines...]
loss 0.0001     x = {0.016149678247001, 0.48861901172629}
got {0.016106419314241, 0.48864949712661} want {0, 0.5}
```

## Graphviz

Calling `Value:graphviz_dot()` will write [graphviz](https://graphviz.org/) dot language displaying the Autodiff graph to the specified output. For example

```
$ lua examples/graphviz.lua | dot -Tsvg -o examples/graph.svg
```

Gives

![](examples/graph.svg)

## MNIST

Since light is scalar based it's unusably slow for mnist. I have a mnist example, but the gradients are hardcoded (and it's only a single layer)

```
$ lua examples/mnist/main.lua
label   5
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . # # # . # # # # . . . . . .
. . . . . . . . # # # # # # # # # # # # # # . . . . . .
. . . . . . # # # # # # # # # # # # # # . . . . . . . .
. . . . . . # # # # # # # # # # . . . . . . . . . . . .
. . . . . . # # # # # # . . . # . . . . . . . . . . . .
. . . . . . . . . # # # . . . . . . . . . . . . . . . .
. . . . . . . . . # # # . . . . . . . . . . . . . . . .
. . . . . . . . . . # # # . . . . . . . . . . . . . . .
. . . . . . . . . . . # # # # . . . . . . . . . . . . .
. . . . . . . . . . . # # # # # . . . . . . . . . . . .
. . . . . . . . . . . . . # # # # . . . . . . . . . . .
. . . . . . . . . . . . . . # # # # . . . . . . . . . .
. . . . . . . . . . . . . . . # # # # . . . . . . . . .
. . . . . . . . . . . . . # # # # # . . . . . . . . . .
. . . . . . . . . . . # # # # # # # . . . . . . . . . .
. . . . . . . . . # # # # # # # # . . . . . . . . . . .
. . . . . . . # # # # # # # # . . . . . . . . . . . . .
. . . . . # # # # # # # # . . . . . . . . . . . . . . .
. . # # # # # # # # # . . . . . . . . . . . . . . . . .
. . # # # # # # # . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . .
[epoch 1] loss 0.916 accuracy 0.134 maxw 0.001
[epoch 2] loss 0.836 accuracy 0.706 maxw 0.004
[epoch 3] loss 0.746 accuracy 0.738 maxw 0.007
[epoch 4] loss 0.661 accuracy 0.754 maxw 0.012
[epoch 5] loss 0.603 accuracy 0.763 maxw 0.016
[epoch 6] loss 0.584 accuracy 0.781 maxw 0.018
[epoch 7] loss 0.559 accuracy 0.775 maxw 0.019
[epoch 8] loss 0.566 accuracy 0.791 maxw 0.022
[epoch 9] loss 0.517 accuracy 0.794 maxw 0.022
[epoch 10] loss 0.526 accuracy 0.820 maxw 0.031
```
