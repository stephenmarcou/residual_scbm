"""
Utility functions for data loading.
"""

import os
import torch
from torch.utils.data import DataLoader

from datasets.CUB_dataset import CUB_CONCEPT_DATASETS, CUB_FAMILY_DATASETS, CUB_LABEL_ROOT, get_CUB_dataloaders
from datasets.CUB_dataset import get_concept_groups_current_space as get_cub_concept_groups_current_space
from datasets.awa2_dataset import get_AWA2_dataloaders
from datasets.awa2_dataset import get_concept_groups_current_space as get_awa2_concept_groups_current_space

from utils.utils import numerical_stability_check


def get_data(config_base, config, gen, log_file=None):
    """
    Parse the configuration file and return the relevant dataset loaders.

    This function parses the provided configuration file and returns the appropriate dataset loaders based on the
    specified dataset type. It supports the CUB family of datasets and AwA2.

    Args:
        config_base (dict): The base configuration dictionary.
        config (dict): The data configuration dictionary containing dataset and data path information.
        gen (object): A generator object to control the randomness of the data loader.

    Returns:
        tuple: A tuple containing the training data loader, validation data loader, and test data loader.
    """
    if config.dataset in CUB_FAMILY_DATASETS:
        print(f"{config.dataset} DATASET")
        trainset, validset, testset = get_CUB_dataloaders(
            config, config_base.incomplete
        )
    elif config.dataset == "AwA2":
        print("AwA2 DATASET")
        trainset, validset, testset = get_AWA2_dataloaders(
            config, config_base.incomplete
        )
    else:
        raise NotImplementedError(
            f"ERROR: Dataset {config.dataset} not supported!"
        )
    config = config_base
    train_loader = DataLoader(
        trainset,
        batch_size=config.model.train_batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=gen,
        drop_last=True,
        persistent_workers=config.workers > 0,
    )
    val_loader = DataLoader(
        validset,
        batch_size=config.model.val_batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=gen,
        persistent_workers=config.workers > 0,
    )
    test_loader = DataLoader(
        testset,
        batch_size=config.model.val_batch_size,
        num_workers=config.workers,
        generator=gen,
    )

    return train_loader, val_loader, test_loader


def get_empirical_covariance(dataloader):
    """
    Compute the empirical covariance matrix of the concepts in the given dataloader.

    This function computes the empirical covariance matrix of the concepts in the given dataloader.
    It first concatenates all the concept data into a single tensor, then applies a logit transformation
    to the data to work in the correct space. The covariance matrix is computed from the transformed data
    and brought into a lower triangular form using Cholesky decomposition. In comments, an alternative
    covariance computation is provided that can be used if the dataset is too large to fit into memory.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader containing batches of data with a "concepts" key.

    Returns:
        torch.Tensor: The lower triangular form of the empirical covariance matrix.
    """
    data = []
    for batch in dataloader:
        concepts = batch["concepts"]
        data.append(concepts)
    print(f"First concept batch: {data[0]}")  # Print the first batch of concepts to check the data format
    data = torch.cat(data)  # Concatenate all data into a single tensor
    data_logits = torch.logit(0.05 + 0.9 * data)
    covariance = torch.cov(data_logits.transpose(0, 1))

    # Bringing it into lower triangular form
    covariance = numerical_stability_check(covariance, device="cpu")
    lower_triangle = torch.linalg.cholesky(covariance)

    ####### Alternative cov computation if dataset was too large for memory
    # num_samples = 0
    # for i, batch in enumerate(dataloader):
    #     concepts = batch["concepts"]
    #     if i == 0:
    #         logits = torch.logit(0.05 + 0.9 * concepts).sum(0)
    #     else:
    #         logits += torch.logit(0.05 + 0.9 * concepts).sum(0)
    #     num_samples += concepts.shape[0]
    # logits_mean = logits / num_samples

    # for i, batch in enumerate(dataloader):
    #     concepts = batch["concepts"]
    #     temp = (torch.logit(0.05 + 0.9 * concepts) - logits_mean).unsqueeze(-1)
    #     if i == 0:
    #         cov = torch.matmul(temp, temp.transpose(-2, -1)).sum(0)
    #     else:
    #         cov += torch.matmul(temp, temp.transpose(-2, -1)).sum(0)
    # cov = cov / num_samples

    ########
    return lower_triangle


def get_concept_groups(config):
    """
    Retrieve the concept groups based on the dataset specified in the configuration.

    This function retrieves the concept groups based on the dataset specified in the configuration.
    This is used for plotting the heatmap of the correlation matrix with the correct concept names.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        list: A list of concept names.
    """
    if config.dataset in CUB_CONCEPT_DATASETS:
        # Oracle grouping based on concept type for CUB
        with open(
            os.path.join(config.data_path, CUB_LABEL_ROOT, "CUB_200_2011/concept_names.txt"),
            "r",
        ) as f:
            concept_names = []
            for line in f:
                concept_names.append(line.replace("\n", "").split("::"))
        concept_names_graph = [": ".join(name) for name in concept_names]

    else:
        concept_names_graph = [str(i) for i in range(config.num_concepts)]

    return concept_names_graph



def get_intervention_concept_groups(config):
    """
    Concept groups for group-wise interventions, indexed in the run's own concept space.

    CUB family: the 28 ATTRIBUTE_PARTS semantic groups of the 112 CBM attributes. AwA2: the
    28 CEM/ECBM groups of the 85 predicates. On an incomplete run (config.incomplete) the
    groups are re-indexed through the split's own record of what was removed (CUB: info.txt,
    AwA2: concept_groups.json), so a group that lost every concept disappears and a partially
    removed one keeps its survivors.

    Returns an ordered {group_name: [concept_idx, ...]} that partitions
    range(config.data.num_concepts). Anything else raises: a group policy that silently
    skipped concepts would give a curve that never reaches full intervention.
    """
    dataset = config.data.dataset
    incomplete = bool(config.get("incomplete", False))
    if dataset in CUB_CONCEPT_DATASETS:
        groups = get_cub_concept_groups_current_space(config.data, incomplete=incomplete)
    elif dataset == "AwA2":
        groups = get_awa2_concept_groups_current_space(config.data, incomplete=incomplete)
    else:
        raise NotImplementedError(
            f"Group-wise interventions need a concept grouping, and none is defined for "
            f"dataset {dataset!r} (available: {CUB_CONCEPT_DATASETS + ('AwA2',)})."
        )

    covered = sorted(idx for idxs in groups.values() for idx in idxs)
    expected = list(range(config.data.num_concepts))
    if covered != expected:
        raise ValueError(
            f"Concept groups for {dataset} must partition the {config.data.num_concepts} "
            f"concepts of this run; they cover {covered} instead. For an incomplete run, "
            f"check that data.pkl_file_dir ({config.data.get('pkl_file_dir')}) is the split "
            f"the model was trained on."
        )
    return groups


def make_analysis_loader(loader, batch_size, num_workers, pin_memory=True):
    """ Do not shuffle or drop last, since we want to analyze all samples. """
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
