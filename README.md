# fission-plugins

Post-processing plugins for [Fission](https://github.com/fission-systems/Fission) pseudocode.

> **Status: design draft.** Nothing here is trained, released or measured yet.
> The working name, license and layout are proposals; see [`docs/DESIGN.md`](docs/DESIGN.md).

A plugin reads one decompiled function (Fission's `decomp --json` output) and returns an *edit list*
(variable and function renames, comments, type hints). It never returns replacement code, and the
applier only rewrites whole identifiers, so a plugin cannot change what the pseudocode computes.

The first plugin, `neural-rename`, is planned to use a fine-tuned, quantized small code model that runs locally.

## Try the skeleton

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

```python
import json
from fission_plugins import EditList, FunctionRecord, apply_edits

row = json.load(open("tests/fixtures/checksum.fission.json"))
record = FunctionRecord.from_fission_json(row)
print(apply_edits(record, EditList(renames={"param_1": "buf", "param_2": "len"})).code)
```

## License

Apache-2.0 (proposed; matches Fission). Model weights are licensed separately in each model card.
