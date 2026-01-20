import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os


class DDPManager:
    def __init__(self, backend="nccl"):
        self._initialized = False
        self.backend = backend
        self.use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        self._setup()

    def _setup(self):
        if self.use_ddp:
            if not dist.is_initialized():
                dist.init_process_group(backend=self.backend, init_method="env://")
            self._initialized = True
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )

        self.is_master = self.rank == 0

        if self.is_master:
            print(
                f"Initialized: "
                f"Distributed={self.is_distributed}, "
                f"World Size={self.world_size}, "
                f"Rank={self.rank}, "
                f"Device={self.device}"
            )

    @property
    def is_distributed(self):
        return self.use_ddp and self._initialized

    def barrier(self):
        if self.is_distributed:
            dist.barrier()

    def cleanup(self):
        if self.is_distributed:
            dist.destroy_process_group()
        if self.is_master:
            print("Group destroyed.")

    def wrap_model(self, model):
        if self.is_distributed:
            return DDP(
                model.to(self.device),
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )
        else:
            return model.to(self.device)


if __name__ == "__main__":
    raise RuntimeError(
        "This file is not meant to be run directly.\n"
        "Import the functions you need instead."
    )
