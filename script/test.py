import torch
from torch.autograd import Variable
from functions.add import IndexedSumFunction
a = Variable(torch.Tensor([0.1, 0.2, 0.3, 0.4]), requires_grad=True)
b = Variable(torch.LongTensor([3, 1]), requires_grad=False)
c = IndexedSumFunction()(a, b)
l = (c[0]*1+c[1]*2)
l.backward()
assert list(a.grad.cpu().data.numpy()) == [1.0, 1.0, 1.0, 2.0]

if torch.cuda.is_available():
    a = Variable(torch.CudaTensor([0.1, 0.2, 0.3, 0.4]), requires_grad=True)
    b = Variable(torch.CudaLongTensor([3, 1]), requires_grad=False)
    c = IndexedSumFunction()(a, b)
    l = (c[0]*1+c[1]*2)
    l.backward()
    assert list(a.grad.cpu().data.numpy()) == [1.0, 1.0, 1.0, 2.0]
