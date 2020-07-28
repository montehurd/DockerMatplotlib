FROM python:3

# set a directory for the app
WORKDIR /usr/src/DockerTest

# copy all the files to the container
COPY . .

# install dependencies
RUN pip3 install numpy
RUN pip3 install matplotlib

# the image writer complains without this set to a temp dir
ENV MPLCONFIGDIR=/tmp/

# tell the port number the container should expose
EXPOSE 8383

# run the command
CMD ["python3", "-m", "http.server", "--cgi", "8383"]
