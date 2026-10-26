"""Training-only level sampling and paired scan-boundary crops."""

import copy
import pickle
from pathlib import Path
import numpy as np


# Crop the CT and mask together and update sample positions.
def crop_pair(data, seg, properties, *, axis, cut, keep_upper):
    size = data.shape[axis + 1]
    if seg.shape[1:] != data.shape[1:]:
        raise ValueError("Invalid paired boundary crop")

    if not 0 < cut < size:
        raise ValueError("Invalid paired boundary crop")
    selection = [slice(None)] * 4
    if keep_upper:
        selection[axis + 1] = slice(cut, None)
    else:
        selection[axis + 1] = slice(None, cut)
    result = copy.deepcopy(properties)
    for level, points in result["class_locations"].items():
        points = np.asarray(points)
        if not len(points):
            continue
        if keep_upper:
            selected = points[:, axis + 1] >= cut
        else:
            selected = points[:, axis + 1] < cut
        points = points[selected].copy()
        if keep_upper:
            points[:, axis + 1] -= cut
        result["class_locations"][level] = points
    result["boundary_crop"] = {
        "axis": axis,
        "cut": int(cut),
        "keep_upper": bool(keep_upper),
    }
    return np.asarray(data[tuple(selection)]), np.asarray(seg[tuple(selection)]), result


class BoundaryTrainingDataset:
    """Sample L1–L5 and negative scans equally, keeping all mask labels."""

    identifiers = ["L1", "L2", "L3", "L4", "L5", "negative"]

    def __init__(self, dataset, directory, axis=0, crop_probability=0.35):
        self.dataset = dataset
        self.axis = axis
        self.crop_probability = crop_probability
        self.groups = {key: [] for key in self.identifiers}
        for case in dataset.identifiers:
            with (Path(directory) / (case + ".pkl")).open("rb") as f:
                properties = pickle.load(f)  # Our own verified preprocessing output.
            present = [
                i for i in range(1, 6) if len(properties["class_locations"].get(i, []))
            ]
            for level in present:
                self.groups[f"L{level}"].append(case)
            if not present:
                self.groups["negative"].append(case)
        if any(not members for members in self.groups.values()):
            raise ValueError("Every level and the negative group need training cases")

    def load_case(self, group):
        case = str(np.random.choice(self.groups[group]))
        if group == "negative":
            level = 0
        else:
            level = int(group[1:])
        return self.prepare_case(case, level)

    def prepare_case(self, case, level, *, allow_crop=True):
        data, seg, previous, properties = self.dataset.load_case(case)
        if previous is not None:
            raise ValueError("Boundary sampler does not support cascades")
        properties = copy.deepcopy(properties)
        properties["source_case_id"] = case
        if not level:
            return data, seg, None, properties
        points = np.asarray(properties["class_locations"][level])
        lower = int(points[:, self.axis + 1].min())
        upper = int(points[:, self.axis + 1].max()) + 1
        # Augment levels away from the existing scan edge; leave real partial cases intact.
        if (
            allow_crop
            and lower > 0
            and upper < data.shape[self.axis + 1]
            and upper - lower >= 8
            and np.random.random() < self.crop_probability
        ):
            retained_fraction = np.random.uniform(0.15, 0.75)
            retained = max(2, int(round((upper - lower) * retained_fraction)))
            keep_upper = bool(np.random.randint(2))
            if keep_upper:
                cut = upper - retained
            else:
                cut = lower + retained
            data, seg, properties = crop_pair(
                data, seg, properties, axis=self.axis, cut=cut, keep_upper=keep_upper
            )
        # This affects foreground patch centring, not the actual multi-class labels.
        properties["class_locations"] = {level: properties["class_locations"][level]}
        return data, seg, None, properties
