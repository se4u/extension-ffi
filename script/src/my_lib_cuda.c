#include <THC/THC.h>
// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;
int indexedsum_forward_cuda(THCudaTensor *input1, THCudaTensor *input2, THCudaTensor *output)
{
  for(unsigned long i=0, counter=0, n=THCudaLongTensor_nElement(increments); i < n; i++){
    long increment = THCudaTensor_fastGet1d(increments, i);
    float val = 0;
    for(long j=0; j < increment; j++)
      val += THTensor_fastGet1d(input1, counter+j);
    THTensor_fastSet1d(output, i, val);
    counter += increment;
  }
  return 1;
}

int indexedsum_backward_cuda(THCudaTensor *grad_output, THCudaTensor *grad_input)
{
  for(unsigned long i=0, counter=0, n=THCudaLongTensor_nElement(increments); i < n; i++){
    long increment = THTensor_fastGet1d(increments, i);
    float val = THTensor_fastGet1d(grad_output, i);
    for(long j=0; j < increment; j++)
      THTensor_fastSet1d(grad_input, counter+j, val);
    counter += increment;
  }
  return 1;
}
