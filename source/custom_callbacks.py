import lightning
import optuna
import utilities

class KillTrainingCallback(lightning.pytorch.callbacks.Callback):
    """Kills current optuna study/trial upon pressing kill bind."""
    def __init__(self, config: object):
        super().__init__()
        self.config = config
        self.must_stop_study = False

    def on_train_batch_end(self, trainer, *args, **kwargs):
        """Pytorch Lightning hook to stop mid-trial."""
        if not self.must_stop_study and utilities.should_kill(self.config):
            self.must_stop_study = True
            trainer.should_stop = True

    def on_validation_batch_end(self, trainer, *args, **kwargs):
        """Pytorch Lightning hook to stop mid-trial."""
        if not self.must_stop_study and utilities.should_kill(self.config):
            self.must_stop_study = True
            trainer.should_stop = True

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Optuna hook: Stops the study after the objective function returns."""
        if self.must_stop_study:
            study.stop()