# functions/add.py
from torch.autograd import Function
from _ext import my_lib
class IndexedSumFunction(Function):
    def forward(self, input1, increment):
        self.save_for_backward(input1, increment)
        output = input1.new(len(increment))
        if input1.is_cuda:
            my_lib.indexedsum_forward_cuda(input1, increment, output)
        else:
            my_lib.indexedsum_forward(input1, increment, output)
        return output
    
    def backward(self, grad_output):
        input1, increment = self.saved_variables
        grad_input = input1.data.new(input1.shape)
        if grad_output.is_cuda:
            my_lib.indexedsum_backward_cuda(grad_output, increment.data, grad_input)
        else:
            my_lib.indexedsum_backward(grad_output, increment.data, grad_input)
        return grad_input, None
