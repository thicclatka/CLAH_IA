import streamlit as st
import base64
import os
from pathlib import Path
from typing import Callable
from CLAH_ImageAnalysis.utils import db_utils
from CLAH_ImageAnalysis.utils import paths
from CLAH_ImageAnalysis.utils import text_dict

EMOJIS = text_dict()["emojis"]
HASH = text_dict()["breaker"]["hash"]


def create_titleWlogo(LOGO_IMAGE: str, TITLE: str):
    """Set the title of the page

    Parameters:
        LOGO_IMAGE (str): Path to the logo image
        TITLE (str): Title of the page
    """
    st.markdown(
        """
    <style>
    .container {
        display: flex;
        gap: 10px;
        align-items: left;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
    }
    .logo-img {
        float:right;
        height: 75px;
        width: 75px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
    <div class="container">
    <img class="logo-img" src="data:image/png ;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text">{TITLE}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def set_title(
    LOGO_IMAGE: str,
    TITLE: str,
    refresh_func: Callable | None = None,
    create_path_db_func: Callable | None = None,
    DB_NAME: str | None = None,
):
    """Set the title of the page

    Parameters:
        TITLE (str): Title of the page
        refresh_func (Callable): Function to refresh the page
        create_db_func (Callable): Function to create the database
    """

    def directory_spinner(no_db: bool = False):
        str4spinner = (
            "Scanning for new directories to add to database..."
            if not no_db
            else "No database found, creating database..."
        )
        with st.spinner(str4spinner):
            path_count, session_count = create_path_db_func()
            st.success(f"Found {path_count} directories")
            if session_count:
                st.success(f"Found {session_count} sessions")

    col1, col2 = st.columns([3, 1])
    with col1:
        title_container = st.container()
        with title_container:
            create_titleWlogo(LOGO_IMAGE, TITLE)
    with col2:
        if DB_NAME is not None:
            if refresh_func is not None:
                if st.button(
                    f"{EMOJIS['epage']} Start from scratch", key="refresh_page"
                ):
                    refresh_func()
            if create_path_db_func is not None:
                if st.button(
                    f"{EMOJIS['refresh']} Refresh Directory Database", key="refresh_db"
                ):
                    directory_spinner()
    st.divider()


def init_SearchSection(DB_NAME: str):
    search_term = create_search_thru_db()
    if search_term:
        print(f"Search term: {search_term}")
        st.divider()
        display_search_resultsNadd_to_pathVar(DB_NAME, search_term)
    if st.session_state.paths:
        st.divider()
        display_selected_paths()
        clear_all_paths()
        st.divider()


def create_search_thru_db():
    st.header("Select Directories to Process")
    st.write(
        """
    Use the search bar to find directories to process. If you cannot find a directory, click "Refresh Directory Database" to update the database.
    """
    )
    search_term = st.text_input(
        "Search directories",
        key="path_search",
        help="Type to search available directories",
        label_visibility="collapsed",
    )
    if st.button("Clear Search", key="clear_search"):
        del st.session_state["path_search"]
        st.session_state["path_search"] = ""
        st.rerun()

    return search_term


def display_search_resultsNadd_to_pathVar(DB_NAME: str, search_term: str):
    if search_term:
        results = db_utils.search_paths_in_db(DB_NAME, search_term)

        if results:
            st.write(f"Found {len(results)} matching directories:")
            st.markdown(
                """
                <style>
                .path-container {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .dotted-line {
                    flex-grow: 1;
                    border-bottom: 2px dotted #ccc;
                    margin: 0 10px;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )
            for idx, path in enumerate(results):
                col_path, col_add = st.columns([1, 1])
                with col_path:
                    st.markdown(
                        f"""
                        <div class="path-container">
                            <span>{path}</span>
                            <div class="dotted-line"></div>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                with col_add:
                    if st.button(f"{EMOJIS['plus']} Add", key=f"add_{idx}"):
                        if path not in st.session_state.paths:
                            st.session_state.paths.append(path)
                            st.success(f"Added: {path}")  # Show confirmation
                            st.rerun()
                        else:
                            st.warning("Already in list!")
        else:
            st.info("No matching directories found")


def display_selected_paths():
    if st.session_state.paths:
        st.write("Selected directories for processing:")
        for idx, path in enumerate(st.session_state.paths):
            col_path, col_remove = st.columns([4, 1])
            with col_path:
                st.text(path)
            with col_remove:
                if st.button(f"{EMOJIS['x']} Remove", key=f"remove_{idx}"):
                    st.session_state.paths.pop(idx)
                    st.success(f"Removed: {path}")  # Show removal confirmation
                    st.rerun()


def clear_all_paths():
    if st.button(f"{EMOJIS['trash']} Clear All Paths"):
        st.session_state.paths = []
        st.rerun()


def init_SessionSelection(DB_NAME: str) -> str:
    selected_session = ""
    if st.session_state.paths:
        st.header("Session Selection")
        st.subheader("1) Select a directory to view sessions:")
        selected_path = st.selectbox(
            "Select a experiment to view sessions:",
            options=st.session_state.paths,
            format_func=lambda x: os.path.basename(x),
            label_visibility="collapsed",
        )
        if selected_path:
            sessions = db_utils.get_sessions_given_path_from_db(DB_NAME, selected_path)
            st.subheader("2) Select a session to view:")
            selected_session = st.radio(
                "Select a session to view:",
                options=sessions,
                format_func=lambda x: os.path.basename(x),
                label_visibility="collapsed",
            )
            st.divider()

    return selected_session


def init_Session2process_selection(DB_NAME: str):
    st.header("Session Selection")
    st.write(
        "By default, all sessions are selected for processing. If you want to modify the session selection, check the box below."
    )
    if st.checkbox("Modify Session Selection?", key="modify_session_selection"):
        if st.session_state.paths:
            st.sidebar.subheader("Session Selector")
            st.sidebar.write("1) Select a directory to view sessions:")
            selected_path = st.sidebar.selectbox(
                "Select a experiment to view sessions:",
                options=st.session_state.paths,
                format_func=lambda x: os.path.basename(x),
                label_visibility="collapsed",
            )
            selected_idx = st.session_state.paths.index(selected_path)
            if selected_path:
                sessions = db_utils.get_sessions_given_path_from_db(
                    DB_NAME, selected_path
                )
                st.sidebar.write(f"Available sessions (Total: {len(sessions)})")
                for idx, session in enumerate(sessions, start=1):
                    st.sidebar.write(
                        f"**{idx:02}**: {os.path.basename(session)}",
                        unsafe_allow_html=True,
                    )
                st.sidebar.write(
                    "2) Enter session range (e.g., '1-3' or '1,2,5') or type 'all' to select all sessions"
                )
                session_range = st.sidebar.text_input(
                    "Enter session range (e.g., '1-3' or '1,2,5') or type 'all' to select all sessions",
                    key=f"range_{selected_path}",
                    value="",
                    label_visibility="collapsed",
                )
                if session_range:
                    st.session_state.sess2process_list[selected_idx] = session_range
    else:
        st.session_state.sess2process_list = ["all"] * len(st.session_state.paths)
    st.write("Current session selection:")
    for path, sess2process in zip(
        st.session_state.paths, st.session_state.sess2process_list
    ):
        st.write(f"{os.path.basename(path)}: {sess2process}")
    st.divider()


def setup_page_config(
    app_name: str,
    app_abbrv: str,
    init_session_state_func: Callable,
    refresh_func: Callable | None = None,
    create_path_db_func: Callable | None = None,
    DB_NAME: str | None = None,
):
    """Setup the page config for the app

    Parameters:
        app_name (str): Name of the app
        app_abbrv (str): Abbreviation of the app
    """
    print(HASH)
    print("STREAMLIT OUTPUT")
    print(HASH)
    print()

    print(f"Setting page config for {app_name}")
    img_dir = Path(paths.get_directory_of_repo_from_file(), "docs/images")
    png_path = Path(img_dir, "png", f"{app_abbrv}_icon.png")
    icon_path = Path(img_dir, "ico", f"{app_abbrv}_icon.ico")
    for p, name in zip(
        [img_dir, png_path, icon_path], ["img_dir", "png_path", "icon_path"]
    ):
        if not p.exists():
            print(f"Path {p} does not exist")
            raise FileNotFoundError(f"Path {name} does not exist")
        else:
            print(f"Path {name} set to: {p}")

    st.set_page_config(
        page_title=f"{app_name}",
        layout="wide",
        page_icon=str(icon_path),
    )

    print(f"Initializing session states for {app_name}")
    init_session_state_func()

    print(f"Setting title for {app_name}")
    set_title(
        LOGO_IMAGE=str(png_path),
        TITLE=app_name,
        refresh_func=refresh_func,
        create_path_db_func=create_path_db_func,
        DB_NAME=DB_NAME,
    )

    return png_path, icon_path
