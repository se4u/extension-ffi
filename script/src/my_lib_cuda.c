#include <THC/THC.h>
extern THCState *state;

int indexedsum_forward_cuda(THCudaTensor *input1, THCudaLongTensor *increments, THCudaTensor *output)
{
  for(unsigned long i=0, counter=0, n=THCudaLongTensor_nElement(state, increments); i < n; i++){
    long increment = THCudaLongTensor_get1d(state, increments, i);
    float val = 0;
    for(long j=0; j < increment; j++){
      val += THCudaTensor_get1d(state, input1, counter+j);
    }
    THCudaTensor_set1d(state, output, i, val);
    counter += increment;
  }
  return 1;
}

int indexedsum_backward_cuda(THCudaTensor *grad_output, THCudaLongTensor* increments, THCudaTensor *grad_input)
{
  for(unsigned long i=0, counter=0, n=THCudaLongTensor_nElement(state, increments); i < n; i++){
    long increment = THCudaLongTensor_get1d(state, increments, i);
    float val = THCudaTensor_get1d(state, grad_output, i);
    for(long j=0; j < increment; j++)
      THCudaTensor_set1d(state, grad_input, counter+j, val);
    counter += increment;
  }
  return 1;
}
