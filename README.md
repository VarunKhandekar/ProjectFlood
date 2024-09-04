# ProjectFlood

This project looks at predicting flood risk on a pixel-by-pixel basis. It primarily uses ConvLSTM architectures, making use of satellite and renalysis data sets, and focuses on Bangladesh.

The project requires the user to set up appropriate environment variables that map to json files. Each JSON file should contain the relevant file paths required. This is the primary way to handle file paths in the code. The user must also set up a Google Earth Engine and Google Drive API and store their credentials. Moreover, the user must gain access to the MSWEP precipitation data set.

The data_extraction folder holds scripts that are used to pull data.
