# DeepHunter Dockerfile
FROM python:3.6-slim

# system dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# copy deephunter repo and fuzzer
COPY Deephunter-backup /app/deephunter
COPY deephunter_fuzzer.py /app/deephunter_fuzzer.py

WORKDIR /app

RUN pip install numpy==1.14.5
RUN pip install h5py==2.8.0
RUN pip install tensorflow==1.5.0
RUN pip install keras==2.1.3
#RUN pip install git+https://github.com/keras-team/keras.git@2.1.3

# other dependencies
RUN pip install \
    matplotlib \
    pillow \
    scikit-learn \
    click \
    future \
    opencv-python==3.4.0.12 \
    tinydb \
    scipy \
    pandas \
    tqdm \
    requests \
    pyflann-py3 \
    xxhash

ENV PYTHONPATH "${PYTHONPATH}:/app/deephunter"

CMD ["python", "/app/deephunter_fuzzer.py"]