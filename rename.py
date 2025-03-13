import os

# Parameters you can modify
folder_path = './drl_trained_nets'  # Path to the folder containing the files
old_prefix = 'drl_multi'       # The current prefix you want to replace
new_prefix = 'comad_v2g'       # The new prefix you want to use

# Rename files in the folder
for filename in os.listdir(folder_path):
    # Check if the file starts with the old prefix
    if filename.startswith(old_prefix):
        # Generate new filename by replacing the prefix
        new_filename = filename.replace(old_prefix, new_prefix, 1)  # Only replace the first occurrence
        # Get full file paths
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {filename} → {new_filename}')

print("Renaming completed.")