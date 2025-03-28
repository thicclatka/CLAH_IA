#!/usr/bin/env python3
import os
from pathlib import Path
import argparse
from CLAH_ImageAnalysis.utils import paths
from CLAH_ImageAnalysis.utils import db_utils


def load_template(systemd_services_dir: Path, template_name: str) -> str:
    """Load a template file from the templates directory"""
    repo_dir = paths.get_directory_of_repo_from_file()
    template_path = Path(repo_dir, "Systemd", "templates", template_name)
    with open(template_path, "r") as f:
        return f.read()


def get_user_input(prompt: str) -> str:
    """Get user input with optional default value"""
    return input(f"{prompt}: ").strip()


def create_service_files(clah_path: str, output_dir: str, service_user: str):
    """Create service files with user input for ports"""

    repo_dir = paths.get_directory_of_repo_from_file()
    dbs_dir = paths.get_path2dbs()
    systemd_services_dir = Path(repo_dir, "SystemdServices")

    # Load templates
    service_template = load_template(systemd_services_dir, "service_template.txt")
    shell_template = load_template(systemd_services_dir, "sh_template.txt")

    # Define the applications with their default settings
    apps = [
        {
            "name": "M2SD_WA",
            "description": "CLAH M2SD App (via Streamlit)",
            "gui_path": f"{repo_dir}/CLAH_ImageAnalysis/GUI/M2SD_WA.py",
            "default_port": 8503,
            "base_url": "m2sd",
        },
        {
            "name": "SD_WA",
            "description": "CLAH Segmentation Dictionary App (via Streamlit)",
            "gui_path": f"{repo_dir}/CLAH_ImageAnalysis/GUI/segDictWA.py",
            "default_port": 8504,
            "base_url": "segdict",
        },
        {
            "name": "CROPPER",
            "description": "CLAH Cropper App (via Streamlit)",
            "gui_path": f"{repo_dir}/CLAH_ImageAnalysis/GUI/Cropper_WA.py",
            "default_port": 8501,
            "base_url": "cropper",
        },
    ]

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSetting up services...")
    print("For each app, you'll be prompted to specify a port number.")
    print("Press Enter to use the default port.\n")

    created_services = []

    for app in apps:
        # Get port number from user
        port = app["default_port"]

        # Create run script
        run_script_path = output_dir / f"run{app['name']}.sh"
        run_script_content = shell_template.format(
            clah_path=clah_path,
            app_name=app["name"],
            gui_path=app["gui_path"],
            port=port,
            base_url=app["base_url"],
        )

        with open(run_script_path, "w") as f:
            f.write(run_script_content)
        os.chmod(run_script_path, 0o755)  # Make executable

        # Create service file
        service_path = output_dir / f"{app['name'].lower()}.service"
        service_content = service_template.format(
            description=app["description"],
            user=service_user,
            working_dir=os.path.dirname(run_script_path),
            script_path=str(run_script_path),
        )

        with open(service_path, "w") as f:
            f.write(service_content)

        created_services.append(
            {"name": app["name"].lower(), "port": port, "service_path": service_path}
        )

    # Print installation instructions
    print("\nService files created successfully!")
    print("\nTo install the services:")
    print("1. Copy the service files to systemd directory:")
    print(f"   sudo cp {output_dir}/*.service /etc/systemd/system/")

    print("\n2. Reload systemd daemon to log changes:")
    print("   sudo systemctl daemon-reload")

    print("\n3. Enable and start each service:")
    for service in created_services:
        print(f"   sudo systemctl enable {service['name']}")
        print(f"   sudo systemctl start {service['name']}")

    print("\n4. Check status of services:")
    for service in created_services:
        print(f"   sudo systemctl status {service['name']}")

    print("\nPorts assigned:")
    for service in created_services:
        print(f"   {service['name']}: {service['port']}")


def main():
    service_user = get_user_input(
        "Enter the user to run the services from (Recommended to choose a user that has admin rights or extensive permissions on the drive holding data.)",
        os.getenv("USER"),
    )

    create_service_files(
        service_user=service_user,
    )


if __name__ == "__main__":
    main()
