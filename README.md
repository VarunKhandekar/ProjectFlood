# ProjectFlood

This project looks at predicting flood risk on a pixel-by-pixel basis. It primarily uses ConvLSTM architectures, making use of satellite and renalysis data sets.

The project requires the user to set up appropriate environment variables that map to json files. This is the primary way to handle file paths. The user must also set up a Google Earth Engine and Google Drive API. Moreover, the user must gain access to the MSWEP precipitation data set.

The data_extraction folder holds scripts that are used to pull data.
