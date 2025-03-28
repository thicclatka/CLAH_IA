import sqlite3
import os
from typing import Callable
from typing import Literal
from pathlib import Path
from tqdm import tqdm
from CLAH_ImageAnalysis.utils import paths
from CLAH_ImageAnalysis.utils import print_done_small_proc
from CLAH_ImageAnalysis.utils import print_wFrame


def create_db_path_with_name(db_name: str) -> Path:
    """
    Create a path to the database with the given name.

    Parameters:
        db_name: Name of database

    Returns:
        Path to the database
    """
    if not db_name.endswith(".db"):
        db_name += ".db"

    db_path = Path(paths.get_path2dbs())
    db_path.mkdir(parents=True, exist_ok=True)
    db_path2use = db_path / db_name
    return db_path2use


def get_search_roots() -> list[str]:
    """
    Get the list of directories to scan from the paths2scan.txt file.

    Returns:
        List of directories to scan
    """
    path2scan = Path(paths.get_path2dbs(), "paths2scan.txt")
    if not path2scan.exists():
        print(f"File {path2scan} not found. Will run db_utils CLI to create it.")
        print("Following the prompts will create the file.")
        try:
            create_txtfiles4db(avoidORscan="scan")
            print(f"File {path2scan} created. Will now proceed...")
        except Exception as e:
            print(f"Error creating file {path2scan}: {e}")
            raise

    with open(str(path2scan), "r") as f:
        search_roots = f.readlines()

    search_roots = [root.strip() for root in search_roots]
    search_roots.sort()
    return search_roots


def get_existing_pathsORsessions(db_name: str, get_paths: bool = True) -> set[str]:
    """Get existing paths or sessions from the database.

    Parameters:
        db_name: Name of database
        get_paths: Whether to get paths or sessions

    Returns:
        Set of existing paths or sessions
    """
    existing_pathsORsessions = set()
    db_path2use = create_db_path_with_name(db_name)
    if db_path2use.exists():
        conn = sqlite3.connect(db_path2use)
        c = conn.cursor()
        c.execute(f"SELECT {'path' if get_paths else 'session'} FROM paths")
        existing_pathsORsessions = set(row[0] for row in c.fetchall())
        conn.close()
    return existing_pathsORsessions


def get_sessions_given_path_from_db(db_name: str, selected_path: str) -> list[str]:
    """Get sessions for a specific path from database.

    Args:
        db_name: Name of database
        selected_path: Path to get sessions for

    Returns:
        List of session names for the selected path
    """
    db_path = create_db_path_with_name(db_name)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT session FROM paths WHERE path = ?", (selected_path,))
    sessions = [row[0] for row in c.fetchall()]

    conn.close()
    return sorted(sessions)


def get_things2avoid() -> list[str]:
    """
    Get the list of things we don't want to look through from the things2avoid.txt file.

    Returns:
        List of things we don't want
    """
    path2things2avoid = Path(paths.get_path2dbs(), "things2avoid.txt")
    if not path2things2avoid.exists():
        print(
            f"File {path2things2avoid} not found. Will run db_utils CLI to create it."
        )
        print("Following the prompts will create the file.")
        try:
            create_txtfiles4db(avoidORscan="avoid")
            print(f"File {path2things2avoid} created. Will now proceed...")
        except Exception as e:
            print(f"Error creating file {path2things2avoid}: {e}")
            raise
    with open(str(path2things2avoid), "r") as f:
        things2avoid = [line.strip() for line in f.readlines()]

    things2avoid = [thing for thing in things2avoid if thing]
    return things2avoid


def thing2avoidcheck(dirpath: str) -> bool:
    """
    Check if the directory is a thing we don't want to look through, which includes hidden files and anything sitting in things2avoid.txt

    Parameters:
        dirpath (str): The directory path to check

    Returns:
        bool: True if the directory is a thing we don't want, False otherwise
    """
    things2avoid = get_things2avoid()
    basename = os.path.basename(dirpath)

    return basename.startswith(".") or any(thing in dirpath for thing in things2avoid)


def remove_hidden_filesNthings2avoid_from_dirnames(dirnames: list[str]) -> list[str]:
    """
    Remove hidden files and things we don't want from the directory names.

    Parameters:
        dirnames: List of directory names

    Returns:
        List of directory names after removing hidden files and things we don't want
    """
    dirnames[:] = [d for d in dirnames if not d.startswith(".")]
    dirnames[:] = [d for d in dirnames if d not in get_things2avoid()]
    return dirnames


def existing_paths_check(existing_paths: set[str], dirpath: str, root: str) -> bool:
    """
    Check if a path is not in the existing_paths set and is a subdirectory of the root directory.

    Parameters:
        existing_paths: Set of existing paths
        dirpath: Path to check
        root: Root directory

    Returns:
        True if the path is not in the existing_paths set and is a subdirectory of the root directory, False otherwise
    """
    oneDir_up = os.path.dirname(dirpath)
    root_check = oneDir_up.startswith(root)
    return (oneDir_up not in existing_paths) and root_check


def add_paths2all_paths(
    all_paths: set[str], dirpath: str, root: str, goUPoneDIR: bool = True
) -> set[str]:
    """
    Add a path to the all_paths set.

    Parameters:
        all_paths: Set of all paths
        dirpath: Path to add
        root: Root directory
        goUPoneDIR: Whether to go up one directory

    Returns:
        Updated set of all paths
    """
    if goUPoneDIR:
        parent_dir = os.path.dirname(dirpath)
    else:
        parent_dir = dirpath

    all_paths.add(parent_dir)
    return all_paths


def search_paths_in_db(db_name: str, search_term: str) -> list[str]:
    """
    Search for paths in the database that contain the search term.

    Parameters:
        db_name: Name of database
        search_term: Term to search for

    Returns:
        List of paths that contain the search term
    """
    db_path2use = create_db_path_with_name(db_name)
    conn = sqlite3.connect(db_path2use)
    c = conn.cursor()

    search_pattern = f"%{search_term}%"
    c.execute("SELECT path FROM paths WHERE path LIKE ? LIMIT 100", (search_pattern,))
    results = [row[0] for row in c.fetchall()]
    conn.close()
    results = list(set(results))
    results.sort()
    return results


def find_paths4dbCreation(
    DB_NAME: str, file_check_func: Callable[[list[str]], bool]
) -> tuple[str, set[str], set[str]]:
    """
    Find paths to add to the database.

    Parameters:
        DB_NAME: Name of database
        file_check_func: Function to check for the file of interest in the directory

    Returns:
        Tuple containing the database path, all paths, existing paths, and new paths, respectively. (db_path2use, all_paths, existing_paths, new_paths)
    """
    db_path2use = str(create_db_path_with_name(DB_NAME))
    search_roots = get_search_roots()
    print(f"Search roots: {search_roots}")

    existing_paths = get_existing_pathsORsessions(DB_NAME, get_paths=True)
    all_paths = existing_paths.copy()

    with tqdm(search_roots, desc="Scanning root directories") as pbar:
        for root in pbar:
            if os.path.exists(root):
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames = remove_hidden_filesNthings2avoid_from_dirnames(dirnames)
                    # if dirpath in existing_paths:
                    #     pbar.set_postfix_str("Skipping due to existing path...")
                    #     continue
                    if thing2avoidcheck(dirpath):
                        pbar.set_postfix_str(
                            "Skipping due to containing thing to avoid..."
                        )
                        continue
                    pbar.set_postfix_str(f"Current: {dirpath}")
                    if file_check_func(filenames):
                        all_paths = add_paths2all_paths(
                            all_paths, dirpath=dirpath, root=root
                        )
    # new_paths = list(all_paths - existing_paths)
    return db_path2use, all_paths, existing_paths


def init_dbCreation(
    DB_NAME: str,
    file_check_func4DB: Callable[[list[str]], bool],
    file_check_func4SessionDict: Callable[[Path], bool],
) -> tuple[int, int]:
    """
    Initialize the database for path and session.

    Parameters:
        DB_NAME (str): Name of database
        file_check_func4DB (Callable[[list[str]], bool]): Function to check for the file of interest in the directory
        file_check_func4SessionDict (Callable[[Path], bool]): Function to check for the file of interest in the session directory

    Returns:
        tuple[int, int]: Total paths and sessions in the database.
    """
    db_path2use, all_paths, existing_paths = find_paths4dbCreation(
        DB_NAME, file_check_func4DB
    )
    if all_paths:
        SessionDict = create_SessionDict(
            all_paths, file_check_func4SessionDict, get_sess_name=True
        )
        add_pathsNsessions2db(DB_NAME, SessionDict)
    else:
        print("No new paths found")

    total_paths = len(get_existing_pathsORsessions(DB_NAME, get_paths=True))
    total_sessions = len(get_existing_pathsORsessions(DB_NAME, get_paths=False))
    print(f"Total paths currently in DB: {total_paths}")
    print(f"Total sessions currently in DB: {total_sessions}")

    return total_paths, total_sessions


def create_SessionDict(
    paths2use: list[str] | set[str],
    file_check_func: Callable[[Path], bool],
    get_sess_name: bool = False,
) -> dict:
    """
    Create session dictionary

    Parameters:
        paths2use: List of paths to use
        file_check_func: Function to check for the file of interest in the session directory
        get_sess_name: Whether to get the session name or the session index

    Returns:
        Session dictionary
    """
    sessionDict = {}
    for path in paths2use:
        if path not in sessionDict.keys():
            sessionDict[path] = []

    for path in paths2use:
        if not os.path.exists(path):
            print(f"Path {path} does not exist. Skipping...")
            continue
        sessions = os.listdir(path)
        sessions = [Path(path, session) for session in sessions]
        sessions.sort()
        for s_idx, session in enumerate(sessions):
            if session.is_dir():
                if file_check_func(session):
                    if get_sess_name:
                        sessionDict[path].append(str(session))
                    else:
                        sessionDict[path].append(s_idx)
    return sessionDict


def add_pathsNsessions2db(db_name: str, SessionDict: dict) -> None:
    """
    Add paths and sessions to the database.

    Parameters:
        db_name: Name of database
        SessionDict: Session dictionary
    """
    db_path2use = create_db_path_with_name(db_name)
    print(f"Creating table in {db_path2use}")
    conn = sqlite3.connect(db_path2use)
    c = conn.cursor()
    print("Clearing table...")
    c.execute("DROP TABLE IF EXISTS paths")
    print("Creating table...")
    c.execute(
        "CREATE TABLE IF NOT EXISTS paths (path TEXT, session TEXT, PRIMARY KEY (path, session))"
    )
    print("Inserting new paths & sessions...")
    for path, sessions in SessionDict.items():
        for sess in sessions:
            c.execute("INSERT OR IGNORE INTO paths VALUES (?, ?)", (path, sess))
    print("Sorting paths & sessions...")
    c.execute(
        """
        CREATE TABLE paths_sorted AS 
        SELECT * FROM paths 
        ORDER BY path ASC, session ASC
    """
    )
    c.execute("DROP TABLE paths")
    c.execute("ALTER TABLE paths_sorted RENAME TO paths")
    conn.commit()
    conn.close()
    print_done_small_proc()


def create_txtfiles4db(avoidORscan: Literal["avoid", "scan"] | None = None) -> None:
    """
    Create text files for avoiding or scanning for db utils.

    Parameters:
        avoidORscan (Literal["avoid", "scan"]): Whether to avoid or scan
    """
    db_path = Path(paths.get_path2dbs())
    path2scan_fname = db_path / "paths2scan.txt"
    path2avoid_fname = db_path / "things2avoid.txt"
    if avoidORscan is None:
        avoidORscan = input("Do you want to avoid or scan? (a/s): ")
        if avoidORscan not in ["a", "s"]:
            print("Invalid input. Please enter 'a' or 's'.")
            return
        else:
            avoidORscan = "avoid" if avoidORscan == "a" else "scan"

    if avoidORscan == "avoid":
        fname2use = path2avoid_fname
    elif avoidORscan == "scan":
        fname2use = path2scan_fname

    current_entries = []
    if fname2use.exists():
        with open(fname2use, "r") as f:
            current_entries = f.readlines()
        current_entries = [entry.strip() for entry in current_entries if entry != ""]
        current_entries.sort()

    things2add = current_entries.copy()

    print(f"{avoidORscan} was selected")
    if current_entries:
        print(f"Current entries in {fname2use}:")
        for entry in current_entries:
            print_wFrame(entry.strip())
    else:
        print("No current entries in this file")

    if avoidORscan == "scan":
        while True:
            print("Please enter a path to scan")
            path_temp = input("Enter path (or type 'done' to finish): ")
            if os.path.exists(path_temp):
                things2add.append(path_temp)
            elif path_temp in things2add:
                print("Path already exists. Please enter a different path.")
            elif path_temp == "done":
                break
            else:
                print("Path does not exist. Check your spelling and try again.")
    elif avoidORscan == "avoid":
        while True:
            print(
                "Please enter a keyword that can be found in folder/file paths to avoid"
            )
            path_temp = input("Enter keyword (type 'done' to finish): ")
            if path_temp == "done":
                if not things2add:
                    things2add = [""]
                break
            else:
                print("Path does not exist. Check your spelling and try again.")

    things2add.sort()

    if things2add == current_entries and things2add[0]:
        print("No new things to add")
    elif things2add != current_entries and things2add[0]:
        print(f"Adding {len(things2add)} things to {fname2use}")
    elif things2add != current_entries and not things2add[0]:
        print(
            "Creating an empty file. If you want to avoid certain directories, add it later."
        )

    with open(fname2use, "w") as f:
        for thing in things2add:
            f.write(f"{thing}\n")

    print_done_small_proc()


if __name__ == "__main__":
    DB_UTILS_str = "DB UTILS "
    bar_hdr = "=" * 40
    bar_ftr = "=" * (len(DB_UTILS_str) + 40)

    print(f"\n{DB_UTILS_str}{bar_hdr}")
    print("Creating txtfiles needed for avoiding or scanning")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avoidORscan", "-as", type=str, choices=["avoid", "scan"], default=None
    )
    args = parser.parse_args()
    create_txtfiles4db(args.avoidORscan)
    print(bar_ftr)
