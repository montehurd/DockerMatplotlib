# Minimal Docker bits serving up web page displaying some Matplotlib graphs

[default.py](https://github.com/montehurd/DockerMatplotlib/blob/master/cgi-bin/default.py) contains 2 methods for making it easier to display [matplotlib](https://matplotlib.org) graphs on a web page:

`base64PNGImageTagForPlot` for static graphs

and

`base64GIFImageTagForAnimation` for animated graphs

[default.py](https://github.com/montehurd/DockerMatplotlib/blob/master/cgi-bin/default.py) also contains example code using these two methods to display a variety of graphs:

![](https://github.com/montehurd/DockerMatplotlib/blob/master/README.gif?raw=true)

## Run it

Ensure you have Docker installed.

Then you can either run it using my Docker Hub tag `montehurd/dockermatplotlib`, or build and run the image locally

### - From Docker Hub

`docker run -p 8383:8383 montehurd/dockermatplotlib`

### - Build and run image locally

Run this command from the `DockerMatplotlib` folder to build the Docker image:

`docker build . -t sometag`

Then this command starts it exposing a simple Python server on port 8383:

`docker run -p 8383:8383 sometag`

## View the web page

Once it's running you can view the sample graphs web page with this URL:

`localhost:8383/cgi-bin/default.py`

## View the page without Docker

The "Docker-ization" of this graphing code was a learning exercise - a variation of the great Docker tutorial found here: https://docker-curriculum.com

Assuming you have Python 3.7 installed, the graphs page may also be served up by a simple Python server w/o Docker:

`python3.7 -m http.server --cgi 8383`

Or you can run this [shell script](https://gist.github.com/montehurd/1f8ffb8de517adc1d54d6e6b62ad9f88).
