import torch


class ReplayBuffer:
    def __init__(self, buffer_size: int, group_size: int):
        assert buffer_size % group_size == 0, "Buffer size must be divisible by group size"
        self.buffer_size = buffer_size
        self.group_size = group_size

        # list[buffer_size][... (tensor for each sample)]
        self.observations: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.done_mask: list[torch.Tensor] = []

    def add(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, done_mask: torch.Tensor) -> None:
        self._add_tensor(self.observations, observations, done_mask)
        self._add_tensor(self.actions, actions, done_mask)
        self._add_tensor(self.rewards, rewards, done_mask)
        self._add_tensor(self.done_mask, done_mask, done_mask)

    def sample(self, sample_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert sample_size % self.group_size == 0, "Sample size must be divisible by group size"

        # Compute sample indices
        sample_size = min(sample_size, len(self.observations))
        groups_sampled = sample_size // self.group_size
        group_indices = torch.randperm(len(self.observations) // self.group_size)[:groups_sampled]
        sample_indices: list[int] = []
        for i in group_indices:
            for j in range(self.group_size):
                sample_indices.append(i * self.group_size + j)

        # Compute the maximum length of the samples
        max_len = 0
        for i in sample_indices:
            max_len = max(max_len, len(self.observations[i]))

        # Sample the tensors
        observations = self._sample_tensor(self.observations, sample_indices, max_len)
        actions = self._sample_tensor(self.actions, sample_indices, max_len)
        rewards = self._sample_tensor(self.rewards, sample_indices, max_len)
        done_mask = self._sample_tensor(self.done_mask, sample_indices, max_len)

        return observations, actions, rewards, done_mask

    def _add_tensor(self, add_to_tensor: list[torch.Tensor], tensor: torch.Tensor, done_mask: torch.Tensor) -> None:
        list_tensor = torch.unbind(tensor, dim=0)
        list_tensor = [t[: torch.sum(done)] for t, done in zip(list_tensor, done_mask, strict=True)]
        add_to_tensor.extend(list_tensor)

        if len(add_to_tensor) > self.buffer_size:
            add_to_tensor = add_to_tensor[-self.buffer_size :]

    def _sample_tensor(self, tensor: torch.Tensor, sample_indices: list[int], max_len: int) -> torch.Tensor:
        sampled_tensor = torch.zeros(len(sample_indices), max_len, *tensor[0].shape)
        for i, index in enumerate(sample_indices):
            sampled_tensor[i, : len(tensor[index])] = tensor[index]
        return sampled_tensor
