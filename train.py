from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from Pos_Former.datamodule import CROHMEDatamodule
from Pos_Former.lit_posformer import LitPosFormer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pathlib import Path
class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint path for the model')
    def before_fit(self):
        if self.config['ckpt_path'] is None:
            cwd = self.trainer.default_root_dir
        else:
            cwd = str(Path(self.config['ckpt_path']).parents[1].absolute())

        checkpoint = ModelCheckpoint(monitor='val_ExpRate', mode='max', save_top_k=1, save_last=True,
                                     filename='{epoch}-{step}-{val_ExpRate:.4f}')
        logger = TensorBoardLogger(cwd, '', '.')
        self.trainer.callbacks.extend([checkpoint])
        self.trainer.logger = logger
        self.trainer.enable_model_summary = True

cli = MyLightningCLI(
    LitPosFormer,
    CROHMEDatamodule,
    save_config_overwrite=True,
    trainer_defaults={"plugins": DDPPlugin(find_unused_parameters=True)},
)