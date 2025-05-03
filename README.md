# Group policy annealing

## Conventions

- Tensor's channels are ordered like this: `(num_groups, group_size, steps, num_actions/num_observations)`. Note that `batch_size = num_groups * group_size`.
- Variables whose type can't be inferred by mypy should be explicitly typed: function signatures, empty list initializations, etc.
