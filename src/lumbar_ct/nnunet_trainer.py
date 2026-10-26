"""The nnU-Net training settings used for the original run and continuations."""

from pathlib import Path
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.compute_initial_patch_size import (
    get_patch_size,
)


class nnUNetTrainerLumbar(nnUNetTrainer):
    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.save_every = 5
        self.oversample_foreground_percent = 0.5
        self.probabilistic_oversampling = True

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation = (-15 / 180 * np.pi, 15 / 180 * np.pi)
        augmented_patch = get_patch_size(
            self.configuration_manager.patch_size,
            rotation,
            rotation,
            rotation,
            (0.7, 1.4),
        )
        # Do not invert superior/inferior or anterior/posterior anatomy.
        self.inference_allowed_mirroring_axes = ()
        return rotation, False, augmented_patch, ()

    def on_epoch_end(self):
        if self.current_epoch + 1 == 25:
            checkpoint_path = Path(self.output_folder) / "checkpoint_midway.pth"
            self.save_checkpoint(str(checkpoint_path))
        super().on_epoch_end()


class nnUNetTrainerLumbarFineTune(nnUNetTrainerLumbar):
    """One bounded continuation using our own complete network weights."""

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 25
        self.initial_lr = 0.001


class nnUNetTrainerLumbarBoundary(nnUNetTrainerLumbar):
    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 15
        self.initial_lr = 0.0001

    # Add boundary sampling to training cases only.
    def get_tr_and_val_datasets(self):
        from .boundary_sampling import BoundaryTrainingDataset
        from .nnunet_inference import tensor_order

        train, validation = super().get_tr_and_val_datasets()
        superior_inferior_axis = tensor_order(self.plans_manager.plans).index(2)
        train = BoundaryTrainingDataset(
            train, self.preprocessed_dataset_folder, axis=superior_inferior_axis
        )
        return train, validation


class nnUNetTrainerLumbarContext(nnUNetTrainerLumbarBoundary):
    """Same boundary trainer, with the larger patch supplied through plans."""

    pass
