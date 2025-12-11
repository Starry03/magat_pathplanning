import sys
import pathlib
import datetime
import argparse


# Ensure project root is on sys.path so top-level packages (dataloader, utils, graphs, ...)
# can be imported when this script is run directly from anywhere
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import AdvancedProfiler
from torch import set_float32_matmul_precision, autograd


from dataloader.Dataloader_dcplocal_notTF_onlineExpert import DecentralPlannerDataLoader
from test.model import PaperArchitecture
from utils.config import process_config


set_float32_matmul_precision("high")


def add_flags(arg_parser: argparse.ArgumentParser) -> None:
    arg_parser.add_argument(
        "config",
        metavar="config_json_file",
        default="None",
        help="The Configuration file in json format",
    )

    arg_parser.add_argument("--mode", type=str, default="train")
    arg_parser.add_argument("--log_time_trained", type=str, default="0")

    arg_parser.add_argument("--num_agents", type=int, default=8)
    arg_parser.add_argument("--map_w", type=int, default=20)
    arg_parser.add_argument("--map_density", type=int, default=1)
    arg_parser.add_argument("--map_type", type=str, default="map")

    arg_parser.add_argument("--trained_num_agents", type=int, default=8)
    arg_parser.add_argument("--trained_map_w", type=int, default=20)
    arg_parser.add_argument("--trained_map_density", type=int, default=1)
    arg_parser.add_argument("--trained_map_type", type=str, default="map")

    arg_parser.add_argument("--nGraphFilterTaps", type=int, default=0)
    arg_parser.add_argument("--hiddenFeatures", type=int, default=0)
    arg_parser.add_argument("--numInputFeatures", type=int, default=128)

    arg_parser.add_argument("--num_testset", type=int, default=4500)
    arg_parser.add_argument("--load_num_validset", type=int, default=200)

    arg_parser.add_argument("--test_epoch", type=int, default=0)
    arg_parser.add_argument("--lastest_epoch", action="store_true", default=False)
    arg_parser.add_argument("--best_epoch", action="store_true", default=False)

    arg_parser.add_argument("--con_train", action="store_true", default=False)
    arg_parser.add_argument("--test_general", action="store_true", default=False)
    arg_parser.add_argument("--train_TL", action="store_true", default=False)
    arg_parser.add_argument("--exp_net_load", type=str, default=None)
    arg_parser.add_argument("--gpu_device", type=int, default=0)

    arg_parser.add_argument("--Use_infoMode", type=int, default=0)
    arg_parser.add_argument("--log_anime", action="store_true", default=False)
    arg_parser.add_argument("--rate_maxstep", type=int, default=2)

    arg_parser.add_argument("--vary_ComR_FOV", action="store_true", default=False)
    arg_parser.add_argument("--commR", type=int, default=7)
    arg_parser.add_argument("--dynamic_commR", action="store_true", default=False)
    arg_parser.add_argument("--symmetric_norm", action="store_true", default=False)

    arg_parser.add_argument("--FOV", type=int, default=9)
    arg_parser.add_argument("--id_env", type=int, default=None)
    arg_parser.add_argument("--guidance", type=str, default="Project_G")

    arg_parser.add_argument("--data_set", type=str, default="")

    arg_parser.add_argument("--update_valid_set", type=int, default=200)
    arg_parser.add_argument("--update_valid_set_epoch", type=int, default=100)

    arg_parser.add_argument("--threshold_SuccessRate", type=int, default=80)
    arg_parser.add_argument("--action_select", type=str, default="soft_max")
    # softmax + max   -- soft_max                       - soft_max
    # nomralize + multinomial                           - sum_multinorm
    # exp + multinomial                                 - exp_multinorm
    arg_parser.add_argument("--nAttentionHeads", type=int, default=0)

    arg_parser.add_argument("--AttentionConcat", action="store_true", default=False)
    arg_parser.add_argument("--test_num_processes", type=int, default=2)
    arg_parser.add_argument("--test_len_taskqueue", type=int, default=4)
    arg_parser.add_argument("--test_checkpoint", action="store_true", default=False)
    arg_parser.add_argument(
        "--test_checkpoint_restart", action="store_true", default=False
    )
    arg_parser.add_argument("--old_simulator", action="store_true", default=False)
    arg_parser.add_argument("--batch_numAgent", action="store_true", default=True)

    arg_parser.add_argument("--no_ReLU", action="store_true", default=False)

    arg_parser.add_argument("--use_Clip", action="store_true", default=False)

    arg_parser.add_argument("--tb_ExpName", type=str, default="")

    arg_parser.add_argument("--attentionMode", type=str, default="GAT_modified")
    # attentionMode
    #               - GAT_origin
    #               - GAT_modified
    #               - KeyQuery
    #               - GAT_DualHead
    #               - GAT_Similarity

    arg_parser.add_argument("--return_attentionGSO", action="store_true", default=False)
    arg_parser.add_argument("--use_dropout", action="store_true", default=False)

    arg_parser.add_argument("--GSO_mode", type=str, default="dist_GSO")
    # GSO_mode
    #               - dist_GSO
    #               - dist_GSO_one      - dist_GSO >0 = 1
    #               - full_GSO          - fully connective graph
    arg_parser.add_argument("--LSTM_seq_len", type=int, default=8)
    arg_parser.add_argument("--LSTM_seq_padding", action="store_true", default=False)
    arg_parser.add_argument("--label_smoothing", type=float, default=0.0)
    arg_parser.add_argument("--bottleneckMode", type=str, default=None)
    # Current Support
    #               attentionMode  -- Key and Query
    #                               BottomNeck_only
    #                               BottomNeck_skipConcat
    #                               BottomNeck_skipConcatGNN
    #                               BottomNeck_skipAddGNN

    arg_parser.add_argument("--bottleneckFeature", type=int, default=128)
    arg_parser.add_argument("--use_dilated", action="store_true", default=False)
    arg_parser.add_argument("--use_dilated_version", type=int, default=1)
    arg_parser.add_argument("--GNNGAT", action="store_true", default=False)
    arg_parser.add_argument("--CNN_mode", type=str, default="Default")
    arg_parser.add_argument("--list_agents", nargs="+", type=int)
    arg_parser.add_argument("--list_map_w", nargs="+", type=int)
    arg_parser.add_argument("--list_num_testset", nargs="+", type=int)

    arg_parser.add_argument("--shuffle_testSet", action="store_true", default=False)

    arg_parser.add_argument("--test_on_ValidSet", action="store_true", default=False)
    arg_parser.add_argument("--list_model_epoch", nargs="+", type=int)
    arg_parser.add_argument(
        "--default_actionSelect", action="store_true", default=False
    )

    arg_parser.add_argument("--load_memory", action="store_true", default=False)


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="")
    add_flags(arg_parser)
    autograd.set_detect_anomaly(True)
    config = process_config(arg_parser.parse_args())
    model = PaperArchitecture(config)
    # model = PaperArchitecture.load_from_checkpoint(
    #     "./tb_logs/2025-10-27 02:26:28.995596/version_0/checkpoints/epoch=56-step=502626.ckpt",
    #     config=config,
    # )
    trainer = Trainer(
        accelerator="gpu",
        # max_epochs=int(config.get("max_epoch", config.get("max_epochs", 10))),
        max_epochs=150,
        precision="16-mixed",
        logger=TensorBoardLogger("tb_logs", name=f"{datetime.datetime.now()}"),
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        benchmark=True,
        profiler=AdvancedProfiler(),
    )
    data_loader = DecentralPlannerDataLoader(config=config)
    if config.get("mode") == "test":
        res = model.test_single(config.get("mode"), data_loader)
        print("RESULT", res)
        return
    model.attach_eval_loaders(
        valid_loader=data_loader.valid_loader,
        test_loader=getattr(data_loader, "test_loader", None),
        training_eval_loader=getattr(data_loader, "test_trainingSet_loader", None),
    )
    trainer.fit(
        model,
        train_dataloaders=data_loader.train_loader,
        val_dataloaders=data_loader.validStep_loader,
    )
    config["mode"] = "test"
    data_loader = DecentralPlannerDataLoader(config=config)
    model.attach_eval_loaders(
        valid_loader=data_loader.valid_loader,
        test_loader=getattr(data_loader, "test_loader", None),
        training_eval_loader=getattr(data_loader, "test_trainingSet_loader", None),
    )
    model.test_single(config.get("mode"), data_loader)


if __name__ == "__main__":
    main()
