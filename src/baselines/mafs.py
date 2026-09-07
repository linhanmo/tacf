from .__init__ import MAFSBaseline as _MAFS, TimeMoEBaseline as _TimeMoE, M2FMoEBaseline as _M2F, PatchTSTBaseline as _PatchTST

MAFS = _MAFS
TimeMoE = _TimeMoE
M2FMoE = _M2F
PatchTST = _PatchTST

__all__ = ["MAFS", "TimeMoE", "M2FMoE", "PatchTST"]
