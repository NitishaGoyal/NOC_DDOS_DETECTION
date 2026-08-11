"""Template only. Do not execute until a frozen checkpoint exists and GPU is idle."""

def build_benchmark_case(device: str, precision: str):
    # Future authorized implementation should:
    # 1. construct the frozen model;
    # 2. load the frozen checkpoint;
    # 3. move model and ONE prepared batch-size-1 input to `device`;
    # 4. set eval() mode;
    # 5. return a zero-argument callable that performs ONLY the complete neural forward.
    #
    # Do not place checkpoint loading, device transfer, decoder, thresholding,
    # archive writing, or metrics inside the callable.
    raise NotImplementedError("Bind only after the relevant frozen checkpoint exists.")
