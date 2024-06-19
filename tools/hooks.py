import operator
import logging
import math
import os

import torch
import torch.distributed as dist

from fvcore.common.checkpoint import Checkpointer

from detectron2.utils.events import get_event_storage
import detectron2.utils.comm as comm
from detectron2.engine.train_loop import HookBase
from detectron2.engine.hooks import BestCheckpointer


class AutoTerminateHook(BestCheckpointer):
    """
    Checkpoints best weights based off given metric.

    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    """
    def __init__(
        self,
        auto_terminate_patience: int, 
        eval_period: int,
        checkpointer: Checkpointer,
        val_metric: str,
        mode: str = "max",
        file_prefix: str = "model_best",
        output_dir: str = "",
    ) -> None:
        """
        A Hook to automatically terminate training when the validation metric is not improved for a while.
        This is based on BestCheckpointer.
        
        Args:
            auto_terminate_patience (int): max eval counts until auto terminate. \
                I.e., if the times of 'current_eval < best_eval' is larger than auto_terminate_patience, terminate training.
            eval_period (int): the period `EvalHook` is set to run.
            checkpointer: the checkpointer object used to save checkpoints.
            val_metric (str): validation metric to track for best checkpoint, e.g. "bbox/AP50"
            mode (str): one of {'max', 'min'}. controls whether the chosen val metric should be
                maximized or minimized, e.g. for "bbox/AP50" it should be "max"
            file_prefix (str): the prefix of checkpoint's filename, defaults to "model_best"
        """
        self._logger = logging.getLogger(__name__)
        self._period = eval_period
        self._val_metric = val_metric
        self.auto_terminate_patience = auto_terminate_patience # max eval counts until auto terminate
        self.patience_counter = 0 # init 
        assert mode in [
            "max",
            "min",
        ], f'Mode "{mode}" to `BestCheckpointer` is unknown. It should be one of {"max", "min"}.'
        if mode == "max":
            self._compare = operator.gt # a > b
            # self._compare = operator.ge # a >= b
        else:
            self._compare = operator.lt # a < b
            # self._compare = operator.le # a <= b
        self._checkpointer = checkpointer
        self._file_prefix = file_prefix
        self.best_metric = None
        self.best_iter = None
        self.output_dir = output_dir

    def _update_best(self, val, iteration):
        if math.isnan(val) or math.isinf(val):
            return False
        self.best_metric = val
        self.best_iter = iteration
        return True

    def _best_checking(self):
        if comm.is_main_process():
            metric_tuple = self.trainer.storage.latest().get(self._val_metric)        
            if metric_tuple is None:
                self._logger.warning(
                    f"Given val metric {self._val_metric} does not seem to be computed/stored."
                    "Will not be checkpointing based on it."
                )
                return
            else:
                latest_metric, metric_iter = metric_tuple
            self._write_eval_result(latest_metric)

            if self.best_metric is None: # first eval
                if self._update_best(latest_metric, metric_iter):
                    additional_state = {"iteration": metric_iter}
                    self._checkpointer.save(f"{self._file_prefix}", **additional_state)
                    self._logger.info(
                        f"Saved first model at {self.best_metric:0.5f} @ {self.best_iter} steps"
                    )
            elif self._compare(latest_metric, self.best_metric): # when current > best
                self.patience_counter = 0 # init again
                additional_state = {"iteration": metric_iter}
                self._checkpointer.save(f"{self._file_prefix}", **additional_state)
                self._logger.info(
                    f"Saved best model as latest eval score for {self._val_metric} is"
                    f"{latest_metric:0.5f}, better than last best score "
                    f"{self.best_metric:0.5f} @ iteration {self.best_iter}."
                )
                self._update_best(latest_metric, metric_iter)
            else: # when current < best
                self.patience_counter += 1
                self._logger.info(
                    f"Not saving as latest eval score for {self._val_metric} is {latest_metric:0.5f}, "
                    f"not better than best score {self.best_metric:0.5f} @ iteration {self.best_iter}."
                )
        
            # check terminate or not 
            if self.patience_counter >= self.auto_terminate_patience:
                self._logger.info("\n\n\n\nAuto Termination at {}, current best {}\n\n\n".format(self.trainer.iter, self.best_metric))   
                self.trainer.is_terminate = torch.Tensor([True]).to(torch.device('cuda')) # switch
            # else:
            #     assert  self.trainer.is_terminate == torch.Tensor([False]).to(torch.device('cuda'))

        if dist.is_initialized() and comm.get_world_size()>1: # share the status of is_terminate among all processes
            comm.synchronize()
            # print('before broadcast: rank={} | is_terminate={}'.format(comm.get_rank(), self.trainer.is_terminate))
            dist.broadcast(self.trainer.is_terminate, src=0)
            # print('after broadcast: rank={} | is_terminate={}'.format(comm.get_rank(), self.trainer.is_terminate))
        

    def _write_eval_result(self, eval_result):
        assert comm.is_main_process(), 'Only main process can write eval result'
        val_info_path = os.path.join(self.output_dir, 'val_info_epoch.txt')
        mode = 'a' if os.path.exists(val_info_path) else 'w'
        cur_epoch = (self.trainer.iter+1) // self.trainer.iter_per_epoch
        with open(val_info_path, mode) as f:
            f.writelines('{}: {}\n'.format(cur_epoch, eval_result))
        

    def after_step(self):
        # same conditions as `EvalHook`
        next_iter = self.trainer.iter + 1
        if (
            self._period > 0
            and next_iter % self._period == 0
            and next_iter != self.trainer.max_iter
        ):
            self._best_checking()

    # NOTE: No Need to conduct after_train
    # def after_train(self):
    #     # same conditions as `EvalHook`
    #     if self.trainer.iter + 1 >= self.trainer.max_iter:
    #         self._best_checking()
    
    
class AutoStepHook(HookBase):
    def __init__(self, scheduler, eval_period, val_metric):
        
        self._logger = logging.getLogger(__name__)
        self._period = eval_period
        self.scheduler = scheduler
        self._val_metric = val_metric
        self._val_value = -torch.ones(1).to(torch.device('cuda')) # init
    
    def eval_step_scheduler(self):
        if comm.is_main_process():
            metric_tuple = self.trainer.storage.latest().get(self._val_metric)        
            if metric_tuple is None:
                self._logger.warning(
                    f"Given val metric {self._val_metric} does not seem to be computed/stored."
                    "Will not be checkpointing based on it."
                )
                return
            else:
                latest_metric, metric_iter = metric_tuple
            self._val_value = torch.Tensor([latest_metric]).to(torch.device('cuda'))
            
        if dist.is_initialized() and comm.get_world_size()>1: 
            comm.synchronize()
            dist.broadcast(self._val_value, src=0)
        
        self._logger.info("Rank {} | val_value {}".format(comm.get_local_rank(), self._val_value))
        self.scheduler.step(self._val_value)
        self._logger.info("Rank {} | Take a Scheduler Step".format(comm.get_local_rank()))
    
    def after_step(self):
        # same conditions as `EvalHook` 
        # self.scheduler.step() is already done in LRScheduler hook       
        next_iter = self.trainer.iter + 1
        if (
            self._period > 0
            and next_iter % self._period == 0
            and next_iter != self.trainer.max_iter
        ):
            self.eval_step_scheduler()

            