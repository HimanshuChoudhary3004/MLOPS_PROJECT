# base image
FROM python:3.8-slim-buster

# upgrade pip
RUN pip install --upgrade pip

# copying all the files to source directory
COPY . /src

# Set working directory
WORKDIR /src

# GIVING PERMISSIONS TO PYTHON FILES
RUN chmod +X docker_train.py

# Install all the dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Trun docker_train.py
ENTRYPOINT [ "python" ]

# run the command
CMD ["docker_train.py"]

