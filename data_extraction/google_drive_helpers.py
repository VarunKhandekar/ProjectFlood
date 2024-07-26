from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import json
import os

def authenticate(config_file_path: str) -> GoogleDrive:
    """
    Authenticate using PyDrive and return a GoogleDrive object.

    Args:
        config_file_path (str): Path to the configuration file containing Google Drive credentials and OAuth paths.

    Returns:
        GoogleDrive: An authenticated GoogleDrive object.
    """
    with open(config_file_path) as config_file:
        config = json.load(config_file)
    google_drive_credentials_path = config['google_drive_credentials']
    google_drive_oauth_path = config['google_drive_oauth']

    gauth = GoogleAuth()
    gauth.LoadClientConfigFile(google_drive_credentials_path)

    # Try to load saved credentials
    gauth.LoadCredentialsFile(google_drive_oauth_path)
    if gauth.credentials is None:
        try:
            gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
        except:
            gauth.CommandLineAuth()  # Use this if LocalWebserverAuth fails
        gauth.SaveCredentialsFile(google_drive_oauth_path)
    elif gauth.access_token_expired:
        gauth.Refresh()
        gauth.SaveCredentialsFile(google_drive_oauth_path)
    else:
        gauth.Authorize()

    return GoogleDrive(gauth)

def send_to_google_drive(drive: GoogleDrive, file_path: str, config_file_path: str, target_location: str, overwrite=True):
    """
    Upload a file to Google Drive, with an option to overwrite if the file already exists.

    Args:
        drive (GoogleDrive): Authenticated GoogleDrive object.
        file_path (str): Path to the file to be uploaded.
        config_file_path (str): Path to the configuration file containing Google Drive folder IDs.
        target_location (str): Target location identifier in the configuration file where the file will be uploaded.
        overwrite (bool): Whether to overwrite the file if it already exists in the target location. Default is True.

    Returns:
        None
    """
    # Get config details
    with open(config_file_path) as config_file:
        config = json.load(config_file)
    folder_id = config[target_location]

    # drive = authenticate(config_file_path)
    file_name = os.path.basename(file_path)

    if overwrite:
        # Search for the file with the same name in the target folder; delete if it exists
        file_list = drive.ListFile({'q': f"title = '{file_name}' and '{folder_id}' in parents and trashed=false"}).GetList()
        for file in file_list:
            file.Delete()

    file = drive.CreateFile({'title': file_name, 'parents': [{'id': folder_id}]})
    file.SetContentFile(file_path)
    file.Upload()
    file = None