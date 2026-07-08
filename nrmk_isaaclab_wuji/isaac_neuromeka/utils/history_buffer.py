import torch


class HistoryBuffer:
    def __init__(self, num_instances, data_dim, max_len=3, device="cuda"):
        self.history_buffer = None  # (num_envs, history_length, data_dim)
        # self.name = name
        self.capacity = max_len
        self.device = device

        self.history_buffer = torch.zeros((num_instances, self.capacity, data_dim), device=self.device)

    def reset(self, env_ids=None):
        self.history_buffer[env_ids] = 0.0

    def reset_with_init_data(self, env_ids, init_data):
        self.history_buffer[env_ids] = init_data[env_ids].unsqueeze(1).repeat(1, self.capacity, 1)

    def __call__(self, current_data):
        # Convert current_data to a torch tensor if it's not already

        # Update history buffer
        self.history_buffer = torch.roll(self.history_buffer, shifts=-1, dims=1)
        self.history_buffer[:, -1] = current_data

    def get_data(self, num_history):
        return self.history_buffer[:, -num_history:].clone()
