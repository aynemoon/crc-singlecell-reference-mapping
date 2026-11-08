"""Static inspection of scvi-tools checkpoint metadata, with no pickle execution.

Only the small data.pkl member is read. GLOBAL, REDUCE, NEWOBJ, BUILD and
persistent tensor references become inert records. No serialized callable,
class, constructor, state setter, or tensor storage is loaded or executed.
Unsupported opcodes fail closed. This is a bounded inspector, not a model loader.
"""

from __future__ import annotations

import pickletools
import zipfile
from dataclasses import dataclass, field
from importlib.metadata import distribution
from pathlib import Path

from scmap.scanvi_knn_analysis import sha256


@dataclass(eq=False)
class Record:
    kind: str
    arguments: object
    state: object = None
    entries: dict = field(default_factory=dict)


def inspect_pickle(data: bytes):
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("checkpoint metadata exceeds 8 MiB inspection limit")
    stack, memo, mark = [], {}, object()

    def marked():
        index = len(stack) - 1
        while index >= 0 and stack[index] is not mark:
            index -= 1
        if index < 0:
            raise ValueError("invalid pickle mark")
        values = stack[index + 1 :]
        del stack[index:]
        return values

    literals = {
        "BINUNICODE",
        "SHORT_BINUNICODE",
        "BINUNICODE8",
        "BINBYTES",
        "SHORT_BINBYTES",
        "BININT",
        "BININT1",
        "BININT2",
        "LONG1",
        "LONG4",
        "BINFLOAT",
    }
    for count, (opcode, arg, _) in enumerate(pickletools.genops(data)):
        if count > 500_000:
            raise ValueError("too many checkpoint metadata opcodes")
        name = opcode.name
        if name in {"PROTO", "FRAME"}:
            continue
        if name in literals:
            stack.append(arg)
        elif name in {"NONE", "NEWTRUE", "NEWFALSE"}:
            stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[name])
        elif name == "MARK":
            stack.append(mark)
        elif name in {"EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE"}:
            stack.append({"EMPTY_DICT": dict, "EMPTY_LIST": list, "EMPTY_TUPLE": tuple}[name]())
        elif name in {"BINPUT", "LONG_BINPUT", "MEMOIZE"}:
            memo[len(memo) if name == "MEMOIZE" else arg] = stack[-1]
        elif name in {"BINGET", "LONG_BINGET"}:
            stack.append(memo[arg])
        elif name == "GLOBAL":
            stack.append(Record("global", arg))
        elif name == "STACK_GLOBAL":
            symbol = stack.pop()
            stack.append(Record("global", f"{stack.pop()} {symbol}"))
        elif name in {"REDUCE", "NEWOBJ"}:
            args, function = stack.pop(), stack.pop()
            stack.append(Record(name, (function, args)))
        elif name == "BUILD":
            state = stack.pop()
            if not isinstance(stack[-1], Record):
                raise ValueError("unsupported BUILD target")
            stack[-1].state = state
        elif name == "BINPERSID":
            stack.append(Record("storage", stack.pop()))
        elif name == "TUPLE":
            stack.append(tuple(marked()))
        elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
            size = int(name[-1])
            values = tuple(stack[-size:])
            del stack[-size:]
            stack.append(values)
        elif name == "APPENDS":
            values = marked()
            if not isinstance(stack[-1], list):
                raise ValueError("unsupported APPENDS target")
            stack[-1].extend(values)
        elif name in {"SETITEM", "SETITEMS"}:
            if name == "SETITEMS":
                values = marked()
            else:
                value, key = stack.pop(), stack.pop()
                values = [key, value]
            target = stack[-1].entries if isinstance(stack[-1], Record) else stack[-1]
            if not isinstance(target, dict) or len(values) % 2:
                raise ValueError("unsupported SETITEM target")
            target.update(zip(values[::2], values[1::2], strict=True))
        elif name == "STOP":
            if len(stack) != 1:
                raise ValueError("invalid pickle final stack")
            return stack[0]
        else:
            raise ValueError(f"unsupported checkpoint opcode: {name}")
    raise ValueError("missing pickle STOP")


def inspect_checkpoint(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if m.filename.endswith("/data.pkl")]
        if len(members) != 1 or members[0].file_size > 8 * 1024 * 1024:
            raise ValueError("expected one bounded data.pkl member")
        root = inspect_pickle(archive.read(members[0]))
    attributes = root["attr_dict"]
    registry = attributes["registry_"]
    fields = registry["field_registries"]
    fields = fields.entries if isinstance(fields, Record) else fields
    labels = fields["labels"]["state_registry"]["categorical_mapping"]
    if not isinstance(labels, Record) or not isinstance(labels.state, tuple):
        raise ValueError("unsupported categorical mapping encoding")
    mapping = labels.state[-1]
    if not isinstance(mapping, list) or not all(isinstance(s, str) for s in mapping):
        raise ValueError("expected a serialized string label mapping")
    unknown = attributes["unlabeled_category_"]
    reference_labels = attributes["labels_"].state[-1]
    # The final linear classifier tensor's rebuild arguments contain its dimensions.
    weight = root["model_state_dict"].entries["classifier.classifier.1.weight"]
    shape = weight.arguments[1][2]
    if registry["model_name"] != "SCANVI" or registry["scvi_version"] != "1.5.1":
        raise ValueError("prediction-order interpretation is pinned to SCANVI 1.5.1")
    if len(set(mapping)) != len(mapping) or mapping[-1] != unknown or shape[0] != len(mapping) - 1:
        raise ValueError("label registry and classifier dimensions disagree")
    framework = distribution("scvi-tools")
    if framework.version != "1.5.1":
        raise ValueError("inspect in the pinned scvi-tools 1.5.1 environment")
    sources = ("scvi/model/base/_training_mixin.py", "scvi/data/fields/_scanvi.py")
    return dict(
        checkpoint_sha256=sha256(path),
        inspection="static pickle opcodes only; no model, tensor or serialized callable loaded",
        scvi_version=registry["scvi_version"],
        model_name=registry["model_name"],
        label_registry=mapping,
        unlabeled_category=unknown,
        classifier_weight_shape=list(shape),
        reference_cells=len(reference_labels),
        reference_unlabelled_cells=reference_labels.count(unknown),
        native_classes=mapping[: shape[0]],
        framework_source_sha256={name: sha256(framework.locate_file(name)) for name in sources},
        mapped_query_runtime_columns_captured=False,
        order_basis="reference registry prefix selected by SCANVI 1.5.1 predict(soft=True)",
        query_order_evidence=(
            "expected unchanged by pinned label transfer; "
            "mapped-query runtime columns not directly captured"
        ),
    )
