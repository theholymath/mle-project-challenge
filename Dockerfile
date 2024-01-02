# Use miniconda3 as the base image
FROM continuumio/miniconda3

# Copy the conda environment file and the create_model.py script to the image
COPY conda_environment.yml /opt/conda_environment.yml
COPY create_model.py /opt/create_model.py
COPY data /opt/data

# Create the conda environment from the file
RUN conda env create -f /opt/conda_environment.yml

# Activate the conda environment and run the create_model.py script
RUN conda run -n housing python /opt/create_model.py

# Save the model as a pickle file
RUN cp model.pkl /opt/model.pkl
