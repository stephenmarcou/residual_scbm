



import ast
import os
from os.path import join

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import wandb



from models.losses import create_loss
from utils.data import get_concept_groups, get_data, make_analysis_loader
from utils.intervention import (
    intervene_cbm,
    intervene_scbm,
    intervene_scbm_residual,
    intervene_scbm_residual_optimized,
    propagates_to_residual,
)
from utils.training import Custom_Metrics, Custom_Regression_Metrics, Target_Metrics, train_one_epoch_cbm, train_one_epoch_scbm, validate_one_epoch_cbm, validate_one_epoch_scbm, validate_one_epoch_scbm_residual, validate_one_epoch_dnn
from utils.utils import reset_random_seeds
import torch
from torch.utils.data import DataLoader
from models.models import create_model
from datasets.CUB_dataset import CUB_CONCEPT_DATASETS, CUB_LABEL_ROOT, CUB_DatasetGenerator, get_CUB_transforms
from datasets.noise import NOISE_TYPES, NoisyImageDataset, split_normalization
from datasets.awa2_dataset import get_incomplete_concept_set_path as get_incomplete_awa2_path

def run(config):
    # Reproducibility
    gen = reset_random_seeds(config.seed)

    # Setting device on GPU if available, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    
    # Additional info when using cuda
    if device.type == "cuda":
        print("Using", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")
        
    
    experiment_path = get_experiment_path(config)   
    
    if not experiment_path.exists():
        raise ValueError(f"Experiment path {experiment_path} does not exist.")
    
    # Epochs setup, need to check this later
    if config.model.training_mode == "joint":
        t_epochs = config.model.j_epochs
    elif config.model.training_mode in ("sequential", "independent"):
        c_epochs = config.model.c_epochs
        t_epochs = config.model.t_epochs
    if config.model.get("p_epochs") is not None:
        p_epochs = config.model.p_epochs
    
    
    
    
    
    # Which split to evaluate / intervene on. 'test' keeps the historical file names, any other
    # split gets its own log so a val run does not overwrite the test results of the same model.
    eval_split = config.inference.get("eval_split", "test")
    if eval_split not in ("test", "val", "train"):
        raise ValueError(
            f"inference.eval_split must be one of ['test', 'val', 'train'], got {eval_split}."
        )
    if config.run_interventions and config.model.model == "dnn":
        raise ValueError(
            "run_interventions=True is not defined for the DNN baseline: it has no concept "
            "bottleneck to intervene on. Use run_inference=True only."
        )
    # Input corruption for post-hoc robustness checks (datasets/noise.py). Applied to the
    # evaluation loader only: the checkpoint is fixed and the intervention strategy is still
    # fitted on the clean train split, so the noisy curve and the clean curve of the same run
    # differ only in the pixels the model sees at evaluation time. None = clean evaluation.
    noise = parse_noise_config(config)
    noise_suffix = f"_noise_{noise['type']}_{noise['amount']:g}" if noise is not None else ""


    # --------------------------------
    # Folder naming to save artifacts and logs
    # -------------------------------
    # 'test' keeps the historical file names; any other split gets its own log so a val run
    # does not overwrite the test results of the same model.
    split_suffix = "" if eval_split == "test" else f"_{eval_split}"
    # Folder the c_mu/res_mu artifacts are dumped to; matches the split's own folder.
    eval_save_folder = eval_split
    # A corrupted evaluation never overwrites the clean logs / artifacts of the same run.
    split_suffix += noise_suffix
    eval_save_folder += noise_suffix
    # Line recorded in both logs so a curve can be traced back to the split it was measured on.
    split_header = f"{eval_split}"
    if noise is not None:
        split_header += (
            f" [noise: {noise['type']}, amount={noise['amount']:g}, seed={noise['seed']}]"
        )

    #-------------------------------
    #   Logs
    #-------------------------------
    if config.run_inference == True:
        log_file_inference = experiment_path / f"inference_log{split_suffix}.txt"
        with open(log_file_inference, "w") as f:
            f.write(f"Inference log for experiment: {experiment_path}\n")
            f.write(f"Evaluation split: {split_header}\n")


    if config.run_interventions == True:
        # Each policy gets its own log, so the group-wise curve of a run never overwrites its
        # concept-wise one ('random' keeps the historical file names).
        inter_policy = config.inference.get("inter_policy", "random")
        policy_suffix = "" if inter_policy == "random" else f"_{inter_policy}"
        # The no-propagation ablation is a second curve on the same checkpoint, split and
        # policy, so everything it writes needs its own name or it would overwrite the standard
        # one. Kept separate from policy_suffix because the per-sample dumps below already carry
        # the policy in their inner folder name, but not this.
        prop_suffix = ""
        if not propagates_to_residual(config):
            if config.model.model != "scbm_residual":
                raise ValueError(
                    "inference.propagate_to_residual=False is only defined for "
                    f"+model=SCBM_RES, not for model {config.model.model!r}: there is no "
                    "residual channel to hold fixed."
                )
            prop_suffix = "_no_res_prop"
        policy_suffix += prop_suffix
        log_file = experiment_path / f"intervention_log{split_suffix}{policy_suffix}.txt"
        with open(log_file, "w") as f:
            f.write(f"Intervention log for experiment: {experiment_path}\n")
            f.write(f"Intervention split: {split_header}\n")
            if not propagates_to_residual(config):
                f.write(
                    "Interventions are NOT propagated to the residual channel: concepts are "
                    "updated through the conditional Gaussian as usual, while the residual Monte "
                    "Carlo draws of the zero-intervention pass are reused unchanged at every "
                    "step.\n"
                )

        # Per-sample predictions behind every point of the curve, next to the log they belong
        # to. The logged curve is an average over the split; this is what lets it be re-read
        # within groups afterwards (on Waterbirds, the four bird x background cells). Off by
        # default: a 200-class sweep dumps ~50MB per run and the standard curve needs none of it.
        intervention_preds_dir = None
        if config.inference.get("save_intervention_predictions", False):
            intervention_preds_dir = (
                experiment_path / f"intervention_preds{split_suffix}{prop_suffix}"
            )
            print(f"Saving per-intervention predictions to {intervention_preds_dir}")



    # Wandb
    os.environ["WANDB_CACHE_DIR"] = os.path.join(
        Path(__file__).absolute().parent, "wandb", ".cache", "wandb"
    )  # S.t. on slurm, artifacts are logged to the right place
    print("Cache dir:", os.environ["WANDB_CACHE_DIR"])
    wandb.init(
        project=config.logging.project,
        reinit=True,
        entity=config.logging.entity,
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.logging.mode,
        tags=[config.model.tag],
    )
    if config.logging.mode in ["online", "disabled"]:
        wandb.run.name = wandb.run.name.split("-")[-1] + "-" + config.experiment_name
    elif config.logging.mode == "offline":
        wandb.run.name = config.experiment_name
    else:
        raise ValueError("wandb needs to be set to online, offline or disabled.")


    
    # ---------------------------------
    #       Prepare data and model
    # ---------------------------------
    
    train_loader, val_loader, test_loader = get_data(
    config,
    config.data,
    gen,
    log_file=log_file_inference if config.run_inference else log_file
    )

    split_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[eval_split]

    if noise is not None:
        # Same records, deterministic test transform with the corruption inserted before
        # normalization (see build_records_loader).
        eval_loader = build_records_loader(config, split_loader.dataset.data, noise=noise)
    else:
        # The train/val loaders are shuffled, so wrap the chosen split in an analysis loader to get a
        # deterministic, complete pass (matters because the saved c_mu/res_mu artifacts are order-dependent).
        # For 'test' this is a no-op: the test loader is already unshuffled with drop_last=False.
        eval_loader = make_analysis_loader(
            split_loader,
            batch_size=config.model.val_batch_size,
            num_workers=config.workers,
        )


    
    # Get concept names for plotting
    concept_names_graph = get_concept_groups(config.data)
   
    
    print(config.data.num_concepts)
    model = create_model(config)
    saved_model_path = experiment_path / "model.pth"
    state_dict = torch.load(saved_model_path, map_location=device)
    print(f"Loaded model state dict from {saved_model_path}")
    model.to(device)
    # Drop checkpoint keys the current model doesn't define (e.g. vestigial params from an
    # older architecture, unused in forward) while still loading strictly against what's left,
    # so a genuinely missing/shape-mismatched key still raises.
    model_keys = set(model.state_dict().keys())
    stale_keys = [k for k in state_dict if k not in model_keys]
    if stale_keys:
        print(f"Ignoring stale checkpoint keys not used by the current model: {stale_keys}")
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(state_dict)
    model.eval()

    if config.model.model == "dnn":
        metrics = Target_Metrics(device, config).to(device)
    elif config.model.regression_task:
        metrics = Custom_Regression_Metrics(config.data.num_concepts, device).to(device)
    else:
        metrics = Custom_Metrics(config.data.num_concepts, device, config).to(device)
    loss_fn = create_loss(config)
    
    
    
    # ---------------------------------
    #       Inference
    # ---------------------------------
    if config.run_inference == True:
        if config.model.model == "dnn":
            validate_one_epoch = validate_one_epoch_dnn
            test_save_kwargs = {"save_concept_meta_data_folder": eval_save_folder}
        elif config.model.model in ("cbm", "cbm_residual"):
            validate_one_epoch = validate_one_epoch_cbm
            test_save_kwargs = {"save_concept_meta_data_folder": eval_save_folder}
        elif config.model.model == "scbm":
            validate_one_epoch = validate_one_epoch_scbm
            test_save_kwargs = {"save_concept_meta_data_folder": eval_save_folder}
        elif config.model.model == "scbm_residual":
            validate_one_epoch = validate_one_epoch_scbm_residual
            test_save_kwargs = {"save_residual_meta_data_folder": eval_save_folder}

        # Still saves concept/residual information given config.data.save_concept_and_residual_channel=true
        print(f"\nEVALUATION ON THE {split_header.upper()} SET:\n")
        validate_one_epoch(
            eval_loader,
            model,
            metrics,
            t_epochs,
            config,
            loss_fn,
            device,
            test=True,
            concept_names_graph=concept_names_graph,
            log_file=log_file_inference,
            **test_save_kwargs,
        )

        # ---------------------------------------------------------
        # Save concept / residual meta data for analysis of concept discovery
        # ---------------------------------------------------------
        # Did inference and saved the c_mu/res_mu artifacts for eval_loader
        # Here we run a full pass over the other splits to save their c_mu/res_mu artifacts too
        # eval_loader is popped from the analysis_loaders dict so it is not re-run
        # We do not do this under noise: the other splits would be dumped clean into the
        # default train/, val/, test/ folders, re-deriving artifacts a clean run already made.
        if config.data.save_concept_and_residual_channel and noise is None:
            save_kwarg = (
                "save_residual_meta_data_folder"
                if config.model.model in ("scbm_residual", "cbm_residual")
                else "save_concept_meta_data_folder"
            )

            # eval_split was already dumped above by the evaluation pass, through an analysis
            # loader with the same deterministic settings, so re-running it here would only
            # duplicate those artifacts at the cost of a full pass.
            analysis_loaders = {
                "val": val_loader,
                "train": train_loader,
                "test": test_loader,
            }
            analysis_loaders.pop(eval_split, None)

            # Same train images, but with the deterministic (test-time) transform instead
            # of the training-time augmentation, so train-vs-test logit comparisons aren't
            # confounded by ColorJitter/RandomResizedCrop/RandomHorizontalFlip.
            if config.data.dataset in CUB_CONCEPT_DATASETS:
                _, test_transform = get_CUB_transforms()
                analysis_loaders["train_with_test_transform"] = DataLoader(
                    CUB_DatasetGenerator(
                        train_loader.dataset.data, transform=test_transform, cache=False
                    ),
                    batch_size=config.model.val_batch_size,
                )

            for folder, base_loader in analysis_loaders.items():
                validate_one_epoch(
                    make_analysis_loader(
                        base_loader,
                        batch_size=config.model.val_batch_size,
                        num_workers=config.workers,
                    ),
                    model,
                    metrics,
                    t_epochs,
                    config,
                    loss_fn,
                    device,
                    test=False,
                    concept_names_graph=concept_names_graph,
                    log_file=log_file_inference,
                    metrics_only_for_saving=True,
                    **{save_kwarg: folder},
                )

    # ---------------------------------
    #       Interventions
    # ---------------------------------

    if config.run_interventions == True:
        if config.model.model in ("cbm", "cbm_residual"):
            intervene = intervene_cbm
        elif config.model.model == "scbm":
            intervene = intervene_scbm
        else:
            #intervene = intervene_scbm_residual
            # CHANGE AFTERWARDS
            intervene = intervene_scbm_residual_optimized
        # Intervention curves
        print(f"\nPERFORMING INTERVENTIONS ON THE {split_header.upper()} SET:\n")
        # train_loader is used for the intervention strategy, for example for empirical percentile
        intervene(
            train_loader, eval_loader, model, metrics, t_epochs, config, loss_fn, device, log_file=log_file,
            save_predictions_dir=intervention_preds_dir,
        )

    wandb.finish(quiet=True)
    return None


def parse_noise_config(config):
    """inference.noise as a plain dict {type, amount, seed}, or None for a clean evaluation."""
    noise_cfg = config.inference.get("noise", None)
    noise_type = noise_cfg.get("type", None) if noise_cfg is not None else None
    if noise_type is None:
        return None
    if noise_type not in NOISE_TYPES:
        raise ValueError(
            f"inference.noise.type must be one of {list(NOISE_TYPES)} or null, got {noise_type!r}."
        )
    if config.data.dataset not in CUB_CONCEPT_DATASETS:
        # The corruption is inserted into the CUB test transform, before its Normalize step.
        raise ValueError(
            f"inference.noise is only implemented for {list(CUB_CONCEPT_DATASETS)}, "
            f"got data.dataset={config.data.dataset}."
        )
    return {
        "type": noise_type,
        "amount": float(noise_cfg.get("amount")),
        "seed": int(noise_cfg.get("seed", 0)),
    }


def build_records_loader(config, records, noise=None):
    """Deterministic loader over CUB-family records with the test-time transform.

    shuffle=False / drop_last=False, so row i of every saved tensor is records[i] and two
    passes over the same records (clean vs noisy) stay sample-aligned. With `noise`, the
    corruption is applied in [0, 1] pixel space between the geometric part of the test
    transform and its Normalize step (datasets/noise.py).
    """
    _, test_transform = get_CUB_transforms()
    if noise is None:
        dataset = CUB_DatasetGenerator(records, transform=test_transform, cache=False)
    else:
        unnormalized, normalize = split_normalization(test_transform)
        dataset = NoisyImageDataset(
            CUB_DatasetGenerator(records, transform=unnormalized, cache=False),
            noise_type=noise["type"],
            amount=noise["amount"],
            seed=noise["seed"],
            normalize=normalize,
        )
    return DataLoader(
        dataset,
        batch_size=config.model.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
    )


def get_experiment_path(config):
    return (
        Path(config.experiment_dir) / config.model.model / config.data.dataset / config.inference.ex_name
    )




def read_run_config(config):
    """Config dict the run was trained with, stored on the first line of its log.txt."""
    experiment_path = get_experiment_path(config)
    with open(os.path.join(experiment_path, "log.txt"), "r") as f:
        return ast.literal_eval(f.readlines()[0])


def update_num_concepts_and_residuals(config, info_line_dict):
    """Restore the channel dimensions the checkpoint was trained with.

    Without this the model is built from the yaml defaults, which only match the
    checkpoint for a complete run trained with the default num_residuals.
    """
    config.data.num_concepts = info_line_dict["data"]["num_concepts"]
    if config.model.model == "dnn":
        # The bottleneck is num_concepts + num_residuals unless the run set an explicit
        # width; both have to be restored or the Linear shapes fail load_state_dict.
        config.model.bottleneck_dim = info_line_dict["model"].get("bottleneck_dim")
    if config.model.model in ("scbm_residual", "cbm_residual", "dnn"):
        if "num_residuals" in info_line_dict["data"]:
            config.data.num_residuals = info_line_dict["data"]["num_residuals"]
        else:
            # .get: the DNN also lands here on datasets whose config has no num_residuals
            # key at all (its bottleneck then defaults to num_concepts, see
            # resolve_bottleneck_dim), and a struct config raises on a plain attribute read.
            print(
                f"Warning: no num_residuals in log.txt, keeping config value "
                f"{config.data.get('num_residuals')}"
            )
    # The encoder architecture is part of the checkpoint's shape (e.g. CIFAR-10 runs use
    # 'simple_CNN' against a 'resnet18' yaml default) and isn't otherwise recovered, so a
    # mismatched default here fails create_model's load_state_dict with missing/unexpected
    # keys and shape mismatches rather than a clear error.
    if "encoder_arch" in info_line_dict["model"]:
        config.model.encoder_arch = info_line_dict["model"]["encoder_arch"]
    else:
        print(
            f"Warning: no encoder_arch in log.txt, keeping config value "
            f"{config.model.encoder_arch}"
        )


# Need to change this function
def update_pkl_dir_and_num_concepts(config):

    experiment_path = get_experiment_path(config)




    with open(os.path.join(experiment_path, "log.txt"), "r") as f:
        lines = f.readlines()
        info_line = lines[0]
        info_line_dict = ast.literal_eval(info_line)

        pkl_file_dir = info_line_dict["data"]["pkl_file_dir"]
        if pkl_file_dir is None:
            # incomplete=True, but the run itself logged no concept-set folder, i.e. it was
            # a complete run. Caught here because the AwA2 default is null, where CUB's is a
            # (never-read) string that would instead fail further down as a missing folder.
            raise ValueError(
                f"Run {config.inference.ex_name} logged no data.pkl_file_dir, so it was "
                f"trained on the complete concept set. Drop incomplete=True."
            )
        config.data.pkl_file_dir = pkl_file_dir.strip("/")

        # Update num concepts and num residuals to create right model
        print(f"info_line_dict: {info_line_dict}")
        update_num_concepts_and_residuals(config, info_line_dict)

    
    # Ensure that the incomplete concept set the run was trained on still exists
    if config.data.dataset in CUB_CONCEPT_DATASETS:
        full_path_pkl_dir = os.path.join(config.data.data_path, CUB_LABEL_ROOT, "incomplete_data", config.data.pkl_file_dir)
        if not os.path.exists(full_path_pkl_dir):
            raise ValueError(f"Pickle directory {full_path_pkl_dir} does not exist.")
    elif config.data.dataset == "AwA2":
        # AwA2 stores its incomplete sets as a reduced predicate matrix under
        # AwA2/<incomplete_dir>/, not as rewritten pkls under CUB_LABEL_ROOT.
        full_path_concept_dir = get_incomplete_awa2_path(config.data)
        if not os.path.exists(full_path_concept_dir):
            raise ValueError(
                f"Incomplete AwA2 concept set {full_path_concept_dir} does not exist."
            )

        
        
    
def check_cluster():
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print("GPU count:", torch.cuda.device_count())
    else:
        print("Using CPU")


def update_config_paths(config):
    """Resolve the data, experiment and pretrained-network roots.

    A cluster run and a workstation run differ only in where these three roots live, so
    each is read from the environment and otherwise keeps whatever the config already
    holds. Set SCBM_DATA_PATH / SCBM_EXPERIMENT_DIR / SCBM_MODEL_DIRECTORY to place the
    datasets, the run folders and the pretrained backbones outside the repository.
    """
    data_path = os.environ.get("SCBM_DATA_PATH")
    experiment_dir = os.environ.get("SCBM_EXPERIMENT_DIR")
    model_directory = os.environ.get("SCBM_MODEL_DIRECTORY")

    if data_path:
        config.data.data_path = data_path
    elif "data_path" not in config.data or config.data.data_path is None:
        # Local data folder if the yaml does not name one
        config.data.data_path = "datasets/"
    if experiment_dir:
        config.experiment_dir = experiment_dir
    if model_directory:
        config.model.model_directory = model_directory



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # Route batched linalg through cuSOLVER rather than MAGMA, whose batched LU faults
    # with "misaligned address" on some matrix sizes (see _inv_triangular in models/losses.py).
    # CUDA-only: a CPU/MPS build has no cuSOLVER and this raises rather than being ignored.
    if torch.cuda.is_available():
        torch.backends.cuda.preferred_linalg_library("cusolver")

    check_cluster()
    update_config_paths(config)
    # Need to change this because it is not incomplete for non-CUB datasets
    if config.incomplete:
        print("Incomplete run")
        update_pkl_dir_and_num_concepts(config)
    else:
        # Complete run: pkl_file_dir is the config default, but the channel
        # dimensions still have to come from the run itself
        update_num_concepts_and_residuals(config, read_run_config(config))

    
    
    
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    run(config)


if __name__ == "__main__":
    main()
