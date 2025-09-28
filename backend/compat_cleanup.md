Shim cleanup summary

This branch removes temporary shim modules that were re-exporting the
implementations now living under `backend.archive.*`. The goal is to reduce
duplication and make the canonical code the archive modules.

Changes performed:

- Converted `backend/data_models.py` into a pure re-export of
  `backend.archive.data_models` (removed duplicate definitions).
- Deleted pure shim files that only re-exported archive modules:
  - `backend/dspy_signatures.py`
  - `backend/main_pipeline.py`

Rationale:
- During a large merge the team introduced short-term shims (via
  `backend.compat.reexport_module`) to avoid breaking imports. Many callers
  have since been migrated to `backend.archive.*`. The shims now add
  duplication and maintenance burden.

Follow-ups:
- Remove or reduce remaining shims after further cleanup.
- Add `pytest.ini` to register custom test marks and reduce warnings.
- Consider a follow-up PR to merge `backend.archive.*` modules into
  top-level `backend/` modules if the team prefers a flatter layout.
