from transformers import Trainer
from transformers.trainer_utils import speed_metrics
import collections, time

class MultiEvalTrainer(Trainer):

    def __init__(self, **args):
        super().__init__(**args)

    def evaluate(
            self,
            eval_dataset = None,
            ignore_keys = None,
            metric_key_prefix: str = ""):
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        all_metrics = {}
        for eval_dataset in self.eval_dataset:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            metric_key_prefix = 'eval_' + eval_dataset.pred_loop_key
            start_time = time.time()

            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
            output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
            all_metrics.update(output.metrics)


        self.log(all_metrics)


        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, all_metrics)
        return all_metrics


