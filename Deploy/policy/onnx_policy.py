# [policy] ONNX 액터 래퍼. 입출력 차원을 계약과 대조하고 유한성 검사.
"""고정 입출력 차원을 검사하는 ONNX Runtime 정책 어댑터."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt



class OnnxPolicy:
    """내보낸 ``[1,obs] -> [1,act]`` 정책 형태를 로드 시점에 검사.

    정책 계약 혼용 방지를 위해 호출자가 입출력 차원 지정.
    """

    def __init__(self, path: str | Path, observation_dim: int, action_dim: int):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError("onnxruntime is required in the MuJoCo environment.") from exc

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"ONNX policy does not exist: {self.path}")
        self.session = ort.InferenceSession(
            str(self.path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected one ONNX input/output, got {len(inputs)}/{len(outputs)}."
            )
        self.input = inputs[0]
        self.output = outputs[0]
        if self.input.type != "tensor(float)" or self.input.shape != [1, self.observation_dim]:
            raise RuntimeError(
                f"Expected float input [1,{self.observation_dim}], got "
                f"{self.input.type} {self.input.shape}."
            )
        if self.output.type != "tensor(float)" or self.output.shape != [1, self.action_dim]:
            raise RuntimeError(
                f"Expected float output [1,{self.action_dim}], got "
                f"{self.output.type} {self.output.shape}."
            )

    def infer(self, observation: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Run one deterministic fixed-batch inference and validate the result."""

        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.observation_dim,):
            raise ValueError(f"Expected observation {(self.observation_dim,)}, got {obs.shape}.")
        if not np.isfinite(obs).all():
            raise ValueError("ONNX observation contains NaN or Inf.")
        result = self.session.run(
            [self.output.name], {self.input.name: np.ascontiguousarray(obs[None, :])}
        )[0]
        if result.shape != (1, self.action_dim) or result.dtype != np.float32:
            raise RuntimeError(f"Unexpected ONNX result: {result.dtype} {result.shape}.")
        action = np.ascontiguousarray(result[0])
        if not np.isfinite(action).all():
            raise RuntimeError("ONNX action contains NaN or Inf.")
        return action

    def describe(self) -> str:
        """Return the exact graph interface for console reports."""

        return (
            f"{self.path}\n"
            f"  input : {self.input.name} {self.input.type} {self.input.shape}\n"
            f"  output: {self.output.name} {self.output.type} {self.output.shape}"
        )
