import sys


def raiseVE_wAllowables(
    input: str | int | float, allowed_values: tuple | set | list, var_type: str
) -> None:
    """
    Raise a ValueError if the input is not in the allowed values

    Parameters:
        input (str | int | float): The input value to check.
        allowed_values (tuple | set | list): The allowed values.
        var_type (str): The type of the variable.
    """
    assert isinstance(allowed_values, (tuple, set, list)), (
        "allowed_values must be tuple, set, or list"
    )
    assert isinstance(var_type, str), "var_type must be a string"
    assert len(allowed_values) > 0, "allowed_values cannot be empty"
    if input not in allowed_values:
        raise ValueError(
            f"Unrecognized {var_type} ({input}). Expected one of {list(allowed_values)}"
        )


def raiseVE_SysExit1(msg: str) -> None:
    """
    Raise a ValueError with a message and exit with status 1

    Parameters:
        msg (str): The message to raise.

    Raises:
        ValueError: If the input is not in the allowed values.
    """

    raise ValueError(msg)
    sys.exit(1)


def BEEP() -> None:
    """Doesn't make a beep, but prints "BEEP!" to the console."""
    print("BEEP!")
