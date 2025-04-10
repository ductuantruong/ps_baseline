import os
import time
import subprocess

# Define the directory to monitor

# Get values from environment variables
input_directory = os.getenv("INPUT_DIRECTORY", "./input/")
output_directory = os.getenv("OUTPUT_DIRECTORY", "./output/")
ckpt_directory = os.getenv("CKPT_DIRECTORY", "./ckpt/")
ckpt_filename = os.getenv("CKPT_FILENAME", "e17_devEER1.163_devmAP0.614.pth")

# Construct full checkpoint path
##ckpt_path="e18_devEER1.438_devmAP0.570.pth"
ckpt_path = os.path.join(ckpt_directory, ckpt_filename)


# Supported audio file extensions
supported_extensions = (".wav", ".mp3", ".flac")

def delete_files_in_dir(directory):
    """Deletes all files inside the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):  # Ensure it's a file, not a subdirectory
            os.remove(file_path)

def process_audio_file(file_path):
    # Command to run the inference script
    command = [
        "python", "inference.py",
        f"--ckpt_path={ckpt_path}",
        f"--data_path={file_path}",
        f"--save_path={output_directory}",
        #f"--score_type_ouput=True",
        f"--consolidate_output=./output/consolidate.json"
    ]
    
    # Run the command
    subprocess.run(command)

def monitor_directory(directory):
    # Keep track of files that have been processed
    processed_files = set()

    while True:
        # List all files in the directory
        for filename in os.listdir(directory):
            # Check if the file has the correct extension and hasn't been processed yet
            if filename.endswith(supported_extensions) and filename not in processed_files:
                file_path = os.path.join(directory, filename)
                print(f"Detected new audio file: {filename}")
                process_audio_file(file_path)
                processed_files.add(filename)

        # Sleep for a while before checking again
        time.sleep(5)

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    delete_files_in_dir(input_directory)

    # Start monitoring the input directory
    monitor_directory(input_directory)

