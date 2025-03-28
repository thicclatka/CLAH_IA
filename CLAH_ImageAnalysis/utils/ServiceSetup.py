import json
import os
from pathlib import Path
from CLAH_ImageAnalysis.utils import paths
from CLAH_ImageAnalysis.utils import db_utils
from CLAH_ImageAnalysis.utils import text_formatting


def load_template(systemd_services_dir: Path, template_name: str) -> str:
    """Load a template file from the templates directory"""
    repo_dir = paths.get_directory_of_repo_from_file()
    template_path = Path(repo_dir, "Systemd", "templates", template_name)
    with open(template_path, "r") as f:
        return f.read()


def get_user_input(prompt: str) -> str:
    """Get user input with optional default value"""
    return input(f"{prompt}: ").strip()


def create_service_files(service_user: str):
    """Create service files with user input for ports"""

    repo_dir = paths.get_directory_of_repo_from_file()
    systemd_services_dir = Path(repo_dir, "SystemdServices")

    str2print = [
        "First need to create files necessary for database creation and path scanning.",
        "You will be prompted to provide a directory/directories to scan for sessions.",
        "You will also be prompted to provide a list of keywords to avoid in the directory/directories.",
        "It is highly recommend to choose non-root directories. Hidden directories are avoided by default.",
    ]
    print(text_formatting.create_multiline_string(str2print))

    db_utils.get_search_roots()
    db_utils.get_things2avoid()

    print("Now ready to create service files.")

    # Load templates
    service_template = load_template(systemd_services_dir, "service_template.txt")
    shell_template = load_template(systemd_services_dir, "sh_template.txt")

    print("Loading app settings...")
    with open(Path(repo_dir, "SystemdServices", "app_settings.json"), "r") as f:
        apps = json.load(f)

    # provide full path to gui_path
    for app in apps.keys():
        apps[app]["gui_path"] = f"{repo_dir}/{apps[app]['gui_path']}"

    # Create output directory if it doesn't exist
    output_dir = Path(repo_dir, "SystemdServices")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Setting up service files...")

    created_services = []

    for app_name, app_settings in apps.items():
        # Create run script
        run_script_path = output_dir / "shell_scripts" / f"run{app_name}.sh"
        run_script_path.parent.mkdir(parents=True, exist_ok=True)

        run_script_content = shell_template.format(
            clah_path=repo_dir,
            app_name=app_name,
            gui_path=app_settings["gui_path"],
            port=app_settings["default_port"],
            base_url=app_settings["base_url"],
        )

        with open(run_script_path, "w") as f:
            f.write(run_script_content)

        # Create service file
        service_path = output_dir / "services" / f"{app_name.lower()}.service"
        service_path.parent.mkdir(parents=True, exist_ok=True)

        service_content = service_template.format(
            description=app_settings["description"],
            user=service_user,
            working_dir=os.path.dirname(run_script_path),
            script_path=str(run_script_path),
        )

        with open(service_path, "w") as f:
            f.write(service_content)

        created_services.append(
            {
                "name": app_name.lower(),
                "port": app_settings["default_port"],
                "service_path": service_path,
            }
        )
    text_formatting.print_done_small_proc()

    # Print installation instructions
    print("\nService files created successfully!")
    print("Services user set to:", service_user)
    print("\nTo install the services:")
    print("1. Copy the service files to systemd directory:")
    print(f"   sudo cp {output_dir}/*.service /etc/systemd/system/")

    print("\n2. Reload systemd daemon to log changes:")
    print("   sudo systemctl daemon-reload")

    print("\n3. Make run scripts executable and enable and start each service:")
    for service in created_services:
        print(f"For {service['name']}:")
        print("    sudo chmod +x run{service['name']}.sh")
        print(f"   sudo systemctl enable {service['name']}")
        print(f"   sudo systemctl start {service['name']}")
        print()

    print("\n4. Check status of services:")
    for service in created_services:
        print(f"   sudo systemctl status {service['name']}")

    print("\nPorts assigned:")
    for service in created_services:
        print(f"   {service['name']}: {service['service_path']}")


def main():
    service_user = input(
        "Enter the user to run the services from (Recommended to choose a user that has admin rights or extensive permissions on the drive holding data): "
    )

    create_service_files(
        service_user=service_user,
    )


if __name__ == "__main__":
    main()
