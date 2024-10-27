from CLAH_ImageAnalysis.utils import print_wFrame, section_breaker
from typing import Any


class CUDA_utils:
    def __init__(self):
        pass

    @staticmethod
    def init_cuda_process():
        """
        Initializes the CUDA process by initializing the CUDA driver and creating a CUDA context.

        Returns:
            context: The CUDA context object.
        """
        import pycuda.driver as cudadrv
        from skcuda.misc import init_context

        cudadrv.init()
        device = cudadrv.Device(0)
        context = init_context(device)
        return context

    @staticmethod
    def close_cuda_process(context: Any) -> None:
        """
        Closes the CUDA process and releases associated resources.

        Parameters:
        - context: The CUDA context to be closed.

        Returns:
        None
        """
        import skcuda.misc as cudamisc

        try:
            cudamisc.done_context(context)  # type: ignore
            cudamisc.shutdown()
        except:
            pass

    def map_parallel(
        self,
        dview: Any,
        funcs_to_parallel: list,
        pars: list,
        a_sync: bool = False,
        cuda: bool = False,
    ) -> Any:
        """
        Maps the given functions to the input parameters in parallel using the specified `dview`.

        Args:
            dview (object): The distributed view object used for parallel execution.
            funcs_to_parallel (list): A list of functions to be parallelized.
            pars (list): A list of input parameters to be passed to the functions.
            a_sync (bool, optional): If True, the mapping is done asynchronously. Defaults to False.
            cuda (bool, optional): If True, CUDA process is closed after mapping. Defaults to False.

        Returns:
            object: The results of the parallel execution.
        """

        print_wFrame(f"See any warning messages/errors between dotted lines:")
        section_breaker("dotted", mini=True)

        if a_sync:
            results = dview.map_async(funcs_to_parallel, pars)
        else:
            results = dview.map(funcs_to_parallel, pars)

        if cuda:
            dview.map(self.close_cuda_process, range(len(pars)))

        section_breaker("dotted", mini=True)
        return results

    @staticmethod
    def has_cuda() -> bool:
        """
        Check if CUDA is available.

        Returns:
            bool: True if CUDA is available, False otherwise.
        """
        try:
            import pycuda.gpuarray as gpuarray
            import pycuda.driver as cudadrv

            return True
        except:
            return False
