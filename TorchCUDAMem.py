
import torch

def b2mb(x): return int(x/2**20)

class TorchTraceMalloc():

    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        torch.cuda.reset_max_memory_allocated() # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        self.end  = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used   = b2mb(self.end-self.begin)
        self.peaked = b2mb(self.peak-self.begin)
        print(f"GPU memory end/peak usage (MB): {self.used:4d}/{self.peaked:4d}")

    def snap_shot(self):
        return [ \
            b2mb( torch.cuda.memory_allocated()     - self.begin ), 
            b2mb( torch.cuda.max_memory_allocated() - self.begin ) ]
