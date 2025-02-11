import gc
import os
import traceback
from contextlib import contextmanager
from functools import wraps
from typing import Any
from typing import Callable
from typing import Generator

from CLAH_ImageAnalysis.utils import print_header
from CLAH_ImageAnalysis.utils import print_wFrame
from CLAH_ImageAnalysis.utils import section_breaker
from CLAH_ImageAnalysis.utils import text_dict
from CLAH_ImageAnalysis.utils import time_utils
from rich import print

text_lib = text_dict()


class ProcessStatusPrinter:
    class output_btw_dots:
        def __init__(
            self,
            pre_msg: str,
            pre_msg_append: bool = False,
            post_msg: list = [],
            style: str = "dotted",
            mini: bool = True,
            done_msg: bool = False,
            timekeep: bool = False,
            timekeep_msg: str | None = None,
            timekeep_seconds: bool = True,
            wFrame: bool = False,
        ) -> None:
            """
            Parameters:
                pre_msg (str): The message to be printed before the dots.
                pre_msg_append (bool): Whether to append the pre_msg to the existing message.
                post_msg (list): The message to be printed after the dots.
                style (str): The style of the dots.
                mini (bool): Whether to print the message in a mini format.
                done_msg (bool): Whether to print the message as a done message.
                timekeep (bool): Whether to timekeep the process.
            """
            self.pre_msg = pre_msg
            self.pre_msg_append = pre_msg_append
            self.post_msg = post_msg
            self.style = style
            self.mini = mini
            self.done_msg = done_msg
            self.timekeep = timekeep
            self.timekeep_msg = timekeep_msg
            self.timekeep_seconds = timekeep_seconds
            self.wFrame = wFrame
            self.print_func = print_wFrame if wFrame else print
            self.TKEEPER = None

        def __call__(self, func: Any) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return wrapper

        def __enter__(self) -> None:
            wrappers = text_lib["wrappers"]

            if self.timekeep:
                self.TKEEPER = time_utils.TimeKeeper(cst_msg=self.timekeep_msg)
            init_msg = (
                f"{self.pre_msg} {wrappers['btw_dots']}"
                if self.pre_msg_append
                else f"{self.pre_msg}"
            )
            self.print_func(init_msg)
            section_breaker(self.style, self.mini)

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            small_proc = text_lib["completion"]["small_proc"]

            section_breaker(self.style, self.mini)
            if self.post_msg:
                self.print_func(self.post_msg)
            if self.done_msg:
                self.print_func(small_proc)
            if self.timekeep:
                self.TKEEPER.setEndNprintDuration(self.timekeep_seconds)
            print()

    @staticmethod
    def _print_process_status(
        start_loop: bool = False,
        end_loop: bool = False,
        end_process: bool = False,
        sess_idx: int = None,
        sess2process: list = None,
        folder_name: str = None,
    ) -> None:
        """
        Prints a message for the end of the current loop or the whole process.

        Parameters:
            start_loop (bool): Whether to print the start of the loop message.
            end_loop (bool): Whether to print the end of the loop message.
            end_process (bool): Whether to print the end of the whole process message.
            sess_idx (int): The index of the session.
            sess2process (list): The list of sessions to process.
            folder_name (str): The name of the folder.
        """

        def _startLoop(sess_idx: int, sess2process: list, folder_name: str) -> None:
            """
            Prints the start of the loop message.

            Parameters:
                sess_idx (int): The index of the session.
                sess2process (list): The list of sessions to process.
                folder_name (str): The name of the folder.
            """
            folder_anlz_str = text_dict()["headers"]["fold_anlz"].format(
                sess_idx + 1, len(sess2process), folder_name
            )
            print_header(folder_anlz_str, subhead=True)

        def _endLoop(sess_idx: int, sess2process: list, folder_name: str) -> None:
            """
            Prints the end of the loop message.

            Parameters:
                sess_idx (int): The index of the session.
                sess2process (list): The list of sessions to process.
                folder_name (str): The name of the folder.
            """
            compl_msg_fLoop = text_dict()["completion"]["forLoop"].format(
                sess_idx + 1, len(sess2process), folder_name
            )
            print_header(compl_msg_fLoop, subhead=True)

        def _endProcess() -> None:
            """
            Prints the end of the for loop message.
            """
            print_header(text_dict()["completion"]["whole_proc"])

        if start_loop:
            _startLoop(sess_idx, sess2process, folder_name)
        elif end_loop:
            _endLoop(sess_idx, sess2process, folder_name)
        elif end_process:
            _endProcess()

    @staticmethod
    @contextmanager
    def print_status_forLoop(
        program_name: str,
        sess_idx: int,
        sess2process: list,
        folder_name: str,
        err_breaker: str = text_lib["breaker"]["hash"],
    ) -> Generator:
        """
        Prints the start and end of the for loop message.

        Parameters:
            program_name (str): The name of the program.
            sess_idx (int): The index of the session.
            sess2process (list): The list of sessions to process.
            folder_name (str): The name of the folder.
            err_breaker (str): The error breaker.

        Returns:
            Generator: A generator that yields the code block to be executed.

        """
        elog_fname = f"zzz_error_log_{program_name}.txt"
        elog_fname = os.path.abspath(elog_fname)
        prog_header = text_lib["headers"][f"main_title_{program_name}"]
        with open(elog_fname, "w") as f:
            f.write(f"Running {prog_header}\n")
            f.write(f"Folder: {folder_name}\n")
            f.write(f"Date: {time_utils.get_current_date_string(wSlashes=True)}\n")
            f.write(
                f"Time Started: {time_utils.get_current_time_string(wColons=True)}\n"
            )
            f.write("See error messages below\n")
            f.write(err_breaker + "\n")
        ProcessStatusPrinter._print_process_status(
            start_loop=True,
            sess_idx=sess_idx,
            sess2process=sess2process,
            folder_name=folder_name,
        )
        try:
            yield
        except Exception as e:
            error_message = traceback.format_exc()
            error_type = type(e).__name__
            last_call = traceback.extract_tb(e.__traceback__)[-1]
            print(f"{error_type} in {last_call.name}: {e}\n{error_message}")
            with open(elog_fname, "a") as f:
                f.write(f"{error_message}\n")
                f.write(err_breaker + "\n")
                f.write(
                    f"Time Error Occurred: {time_utils.get_current_time_string(wColons=True)}\n"
                )
                print()
        else:
            with open(elog_fname, "a") as f:
                f.write("No errors.\n")
                f.write(err_breaker + "\n")
                f.write(
                    f"Time Ended: {time_utils.get_current_time_string(wColons=True)}\n"
                )
        finally:
            ProcessStatusPrinter._print_process_status(
                end_loop=True,
                sess_idx=sess_idx,
                sess2process=sess2process,
                folder_name=folder_name,
            )

    @staticmethod
    def print_end_wholeProc(program_name: str, steps: bool = False) -> Callable:
        """
        Prints the end of the whole process message.

        Parameters:
            program_name (str): The name of the program.
            steps (bool): Whether to print the steps.

        Returns:
            Callable: A decorator that prints the end of the whole process message.
        """

        prog2print = text_lib["headers"][f"main_title_{program_name}"]
        steps2print = text_lib["steps"][f"{program_name}"]["main"] if steps else None

        def decorator(func):
            def wrapper(*args, **kwargs):
                print_header(prog2print, steps2print)
                try:
                    result = func(*args, **kwargs)
                finally:
                    ProcessStatusPrinter._print_process_status(end_process=True)
                return result

            return wrapper

        return decorator

    @staticmethod
    def get_user_confirmation(prompt: str) -> bool:
        """
        Gets user confirmation for a given prompt.

        Parameters:
            prompt (str): The prompt to get user confirmation for.

        Returns:
            bool: True if the user confirms, False otherwise.
        """
        prompt = f"{prompt} [yes/no]: "
        while True:
            user_input = input(prompt)
            if user_input.lower() in text_lib["YES_RESP"]:
                return True
            elif user_input.lower() in text_lib["NO_RESP"]:
                return False
            else:
                print("Invalid input. Please enter yes or no")

    @contextmanager
    def garbage_collector() -> Generator:
        """
        A context manager that collects garbage after the code block is executed.

        Returns:
            Generator: A generator that yields the code block to be executed.
        """
        try:
            yield
        finally:
            gc.collect()
