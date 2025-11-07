from detectron2.engine.hooks import HookBase
from tqdm.auto import tqdm
import torch


class TqdmProgressHook(HookBase):
    def __init__(self):
        self.pbar = None

    def before_train(self):
        self.pbar = tqdm(
            total=self.trainer.max_iter,
            desc="Training",
            initial=self.trainer.start_iter,
            dynamic_ncols=True,
            ncols=150
        )

    def after_step(self):
        storage = self.trainer.storage
        self.pbar.update(1)

        if self.trainer.iter % 20 == 0:
            postfix = {
                'total_loss': f"{storage.history('total_loss').avg(20):.3f}",
                'cls': f"{storage.history('loss_cls').avg(20):.4f}",
                'box': f"{storage.history('loss_box_reg').avg(20):.5f}",
                'rpn_cls': f"{storage.history('loss_rpn_cls').avg(20):.4f}",
                'rpn_loc': f"{storage.history('loss_rpn_loc').avg(20):.4f}",
                'lr': f"{storage.history('lr').latest():.2e}",
                'mem': f"{torch.cuda.max_memory_allocated() / 1024**2:.0f}M",
                'data_time': f"{storage.history('data_time').avg(20):.4f}"
            }

            if 'time' in storage._history:
                postfix['time'] = f"{storage.history('time').avg(20):.4f}"

            self.pbar.set_postfix(postfix, refresh=True)

    def after_train(self):
        if self.pbar:
            self.pbar.close()
