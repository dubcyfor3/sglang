from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.algorithm.trace_recorder import TraceRecorder
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.server_args import ServerArgs


class DllmAlgorithm:

    def __init__(
        self,
        config: DllmConfig,
    ):
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        # Initialize trace recorder if enabled
        self.trace_recorder = TraceRecorder(
            enabled=config.enable_dllm_trace,
            output_dir=config.dllm_trace_output_dir,
        )

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)
