import os
import pickle
import zipfile
from collections import OrderedDict, defaultdict

import numpy as np
import pytest

from scmap.checkpoint_metadata import Record, inspect_checkpoint, inspect_pickle


def forbidden_tensor_constructor(*args):
    raise AssertionError("serialized constructor executed")


class SerializedTensor:
    def __reduce__(self):
        return forbidden_tensor_constructor, (None, 0, (2, 128))


def test_extracts_actual_registry_order_without_loading_objects(tmp_path):
    registry = dict(
        model_name="SCANVI",
        scvi_version="1.5.1",
        field_registries=defaultdict(
            dict,
            labels=dict(
                state_registry=dict(
                    categorical_mapping=np.array(["z", "a", "Unknown"], dtype=object)
                )
            ),
        ),
    )
    content = dict(
        attr_dict=dict(
            registry_=registry,
            unlabeled_category_="Unknown",
            labels_=np.array(["z", "a", "Unknown"], dtype=object),
        ),
        model_state_dict=OrderedDict({"classifier.classifier.1.weight": SerializedTensor()}),
    )
    path = tmp_path / "model.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/data.pkl", pickle.dumps(content, protocol=2))
    report = inspect_checkpoint(path)
    assert report["native_classes"] == ["z", "a"]
    assert report["classifier_weight_shape"] == [2, 128]
    assert report["reference_cells"] == 3 and report["reference_unlabelled_cells"] == 1


def test_serialized_code_is_never_executed(tmp_path):
    marker = tmp_path / "must_not_exist"

    class Payload:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    value = inspect_pickle(pickle.dumps(Payload(), protocol=2))
    assert isinstance(value, Record)
    assert value.kind == "REDUCE"
    assert not marker.exists()


def test_unsupported_opcode_fails_closed():
    with pytest.raises(ValueError, match="unsupported checkpoint opcode"):
        inspect_pickle(b"\x80\x02Pexternal\n.")
