# python import
import os
import sys
import logging
from datetime import datetime
# package import
import torch
import hydra
import pandas as pd
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import TensorBoardLogger
# local import
from module.stage1_train_module import TrainModule
from module.monai_data_module import MonaiDataModule
from custom import callbacks as custom_callbacks
from utils.load_module import get_unique_attr_across
from utils.logger import log_exception
# extra import
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)

@hydra.main(
    version_base="1.2",
    config_path=os.getenv('CONFIGS_LOCATION', 'config'),
    config_name="config",
)
@log_exception(logger=logger)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # print the config
    script = os.path.basename(sys.argv[0])
    script_name = os.path.splitext(script)[0]
    args = sys.argv[1:]
    conda_env = os.getenv('CONDA_DEFAULT_ENV', 'N/A')
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logger.info(f"Script Name: {script_name}")
    logger.info(f"Arguments: {args}")
    logger.info(f"Conda Environment: {conda_env}")
    logger.info(f"Start Time: {start_time}")
    logger.info(f"Training Fold: {cfg.get('dataset_folder').get('dataset').get('fold')}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # set seed
    seed = cfg.get("dataset_folder").get('dataset').get("seed")
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # build trainer
    if HydraConfig.get().mode == hydra.types.RunMode.RUN:
        work_dir = HydraConfig.get().run.dir
    else:
        work_dir = os.path.join(HydraConfig.get().sweep.dir, HydraConfig.get().sweep.subdir)
    tb_logger = TensorBoardLogger(save_dir=work_dir)
    # if you want to use your own callbacks, you can add them to custom_callbacks
    callback_lt = get_unique_attr_across([pl.callbacks, custom_callbacks],
                                         cfg.get("model_folder").get("callbacks"))
    trainer_config = cfg.get("model_folder").get('trainer')
    trainer = pl.Trainer(
        **trainer_config,
        logger=tb_logger,
        callbacks=callback_lt,
    )
    logger.info("trainer built.")

    # datamodule
    data_module = MonaiDataModule(cfg.get("dataset_folder").get("dataset"), cfg.get("dataset_folder").get("dataloader"),
                                  cfg.get("dataset_folder").get("mixup"))
    logger.info("data module built.")

    # graph dataset
    # build one hop graph
    # df = data_module.train_data[['spot_num', 'tissue_x_coords', 'tissue_y_coords']]
    # df = df.assign(spot_num=df['spot_num'].astype(int)).groupby('spot_num', as_index=False).agg(
    #     {'tissue_x_coords': list, 'tissue_y_coords': list})
    # from sklearn.neighbors import kneighbors_graph
    # import numpy as np
    # def per_spot_adjacency(df_row):
    #     x = np.asarray(df_row['tissue_x_coords'])
    #     y = np.asarray(df_row['tissue_y_coords'])
    #     pts = np.column_stack([x, y])          # (n, 2)
    #     adj = kneighbors_graph(pts, n_neighbors=min(8, len(pts)-1), mode='connectivity', include_self=False)
    #     return adj.toarray()
    # adj_list = df.apply(per_spot_adjacency, axis=1).tolist()
    # adj_list = [torch.tensor((adj + adj.T) > 0).fill_diagonal_(1) for adj in adj_list]
    # herarchical graph

    model_config, criterion_config = cfg.get("model_folder").get("model"), cfg.get("model_folder").get("criterion")
    optimizer_config, lr_scheduler_config = cfg.get("model_folder").get("optimizer"), cfg.get("model_folder").get("lr_scheduler", {})
    extra_config = cfg.get("model_folder").get("extra", {})
    if cfg.get('num_classes', False):
        logger.info("Overriding num_classes in model config with cfg.num_classes")
    else:
        cfg['num_classes'] = 2
        logger.info("Setting cfg.num_classes to 2 as default")
    model = TrainModule(cfg.get('num_classes'), **model_config, **criterion_config,
                        **optimizer_config, **lr_scheduler_config, **extra_config)
    logger.info("model built.")

    if cfg.get('train', True):
        trainer.fit(model, data_module, ckpt_path=cfg.get("ckpt_path", None))
        logger.info("training finished.")

    # test part
    if cfg.get('train', True):
        test_ckpt_dir = trainer.logger.log_dir
    else:
        test_ckpt_dir = os.path.join(cfg.get("ckpt_path", ""), "lightning_logs", "version_0")
    if not os.path.exists(test_ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory {test_ckpt_dir} does not exist.")

    ckpt_path = os.path.join(test_ckpt_dir, 'checkpoints')
    version_number = os.path.basename(test_ckpt_dir).split('_')[-1]

    for v, ckpt_file in enumerate(os.listdir(ckpt_path)):
        # Create a new TensorBoardLogger with a specific version
        tb_logger = TensorBoardLogger(save_dir=work_dir, version=f'version_{version_number}_test_{v}')
        test_trainer_cfg = trainer_config.copy()
        test_trainer_cfg['max_epochs'] = 30000  # if test needs train simple model
        # Update the trainer with the new logger
        test_trainer = pl.Trainer(
            **test_trainer_cfg,
            logger=tb_logger,
            callbacks=callback_lt,
        )

        result = test_trainer.test(model, data_module, ckpt_path=os.path.join(ckpt_path, ckpt_file))
        base_name = os.path.basename(ckpt_file)
        pd.DataFrame(result).to_csv(os.path.join(test_trainer.logger.log_dir, f'{base_name}_result.csv'))
        logger.info(f"testing finished for {base_name}.\n\n\n\n\n")

if __name__ == "__main__":
    main()
