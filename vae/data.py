"""MNIST loading and the train/validation/test split required by the task.

Task 1 specifies:

* The dataset must contain 60.000 training images and 10.000 testing images.
* Pixel values are normalised to the range ``[0, 1]`` (by dividing by 255);
  ``torchvision``'s :class:`~torchvision.transforms.ToTensor` does exactly this.
* The 60.000 training images are split so that the *first* 50.000 are used for
  training and the *last* 10.000 are used for validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

# Image geometry of MNIST. ``D`` (the number of features per image) is used
# throughout the assignment, e.g. in the product over ``d = 1 ... D``.
IMAGE_CHANNELS = 1
IMAGE_SIZE = 28
NUM_FEATURES = IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE  # D = 784

_EXPECTED_TRAIN = 60_000
_EXPECTED_TEST = 10_000
_NUM_TRAIN = 50_000  # first 50.000 images -> training
_NUM_VAL = 10_000     # last 10.000 images  -> validation


@dataclass
class MnistData:
    """The three dataset splits plus convenient data loaders."""

    train_set: Dataset
    val_set: Dataset
    test_set: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def load_mnist(
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 0,
    download: bool = True,
) -> MnistData:
    """Load MNIST and build the splits described in Task 1.

    ``ToTensor`` maps the raw ``uint8`` pixels in ``[0, 255]`` to floats in
    ``[0, 1]``, which is exactly the normalisation the task asks for.
    """

    transform = transforms.ToTensor()

    full_train = MNIST(root=root, train=True, download=download, transform=transform)
    test_set = MNIST(root=root, train=False, download=download, transform=transform)

    # Verify the dataset sizes the assignment expects before doing anything else.
    if len(full_train) != _EXPECTED_TRAIN:
        raise ValueError(
            f"Expected {_EXPECTED_TRAIN} training images, got {len(full_train)}."
        )
    if len(test_set) != _EXPECTED_TEST:
        raise ValueError(
            f"Expected {_EXPECTED_TEST} testing images, got {len(test_set)}."
        )

    # Deterministic split: the *first* 50.000 for training, the *last* 10.000
    # for validation. Using contiguous index ranges (rather than a random
    # split) keeps the split reproducible across runs without a seed.
    train_set = Subset(full_train, range(0, _NUM_TRAIN))
    val_set = Subset(full_train, range(_NUM_TRAIN, _NUM_TRAIN + _NUM_VAL))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return MnistData(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
