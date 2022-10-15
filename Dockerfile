FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

# Install torchdiffeq
RUN pip install torchdiffeq imageio einops matplotlib