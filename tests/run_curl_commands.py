import subprocess

# Read the curl commands from a text file
with open("curl_commands.txt", "r") as file:
    curl_commands = file.readlines()

# Execute the curl commands
for command in curl_commands:
    subprocess.run(command, shell=True)
