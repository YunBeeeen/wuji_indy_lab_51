import os
import copy
import torch


class _OnnxBaseMLPPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, input_dim, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.input_dim = input_dim
        self.actor = copy.deepcopy(actor_critic.actor)

    def forward(self, x):
        return self.actor(x)

    def export(self, path, filename, device):
        self.to(device)
        self.eval()

        obs = torch.zeros(1, self.input_dim).to(device)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def export_policy_as_onnx(actor_critic: object, input_dim: int, Exporter: object,
                          path: str, filename="policy.onnx", verbose=False, device="cuda"):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = Exporter(actor_critic, input_dim, verbose)
    policy_exporter.export(path, filename, device)