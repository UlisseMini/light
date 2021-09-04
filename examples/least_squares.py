import torch

def least_squares(A, b):
  # Minimize ||Ax - b||^2
  def cost(x):
    diff = (A.matmul(x) - b)
    mag_squared = diff.dot(diff)
    return mag_squared

  lr = 0.01
  x = torch.zeros(b.size(), requires_grad=True)

  for i in range(1, 1000):
    loss = cost(x)
    print(('loss %.4f\tx = %s') % (loss.data, x))
    x.grad = None # zero grad before backward
    loss.backward()

    with torch.no_grad():
      x -= lr * x.grad

  return x


A = torch.Tensor([
  [1,2],
  [3,4],
])

b = torch.Tensor([1, 2])
x_star = torch.Tensor([0, 0.5]) # solution of Ax = b

x = least_squares(A, b)
print(('got %s want %s') % (x, x_star))


