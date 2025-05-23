# Group policy annealing

## Conventions

- Tensor's channels are ordered like this: `(num_groups, group_size, steps, num_actions/num_observations)`. Note that `batch_size = num_groups * group_size`.
- Variables whose type can't be inferred by mypy should be explicitly typed: function signatures, empty list initializations, etc.
- Ensure that linting and tests succeed: `pre-commit run -a && pytest`

## Installation

- Swig needs to be installed on a system level (eg. `apt install swig` or `pacman -S swig`)
- After that, you need to install Pytorch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) and requirements (`pip install -r requirements.txt`)
