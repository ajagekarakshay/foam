import wandb

__all__ = (
    'WandbRL'
)

class WandbRL:
    def __init__(self, mode="online", **kwargs):
        if wandb.run:
            wandb.finish() # End previous run
        self.experiment = wandb.init(mode=mode, **kwargs)

    def log_metrics(self, metrics, step):
        self.experiment.log(metrics, step=step)
    
    def log_histogram(self, data, step, bins):
        for key, item in data.items():
            self.experiment.log( {key:wandb.Histogram(item, num_bins=bins)},
                                 step=step )

    def log_model(self, data:dict):
        pass
    
    def log_image(self, data, step):
        raise NotImplementedError
    