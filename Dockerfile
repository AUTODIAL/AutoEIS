FROM ubuntu:18.04

ARG PYTHON_VERSION=3.8.5
ARG JULIA_VERSION=1.7.1

ENV container docker
ENV DEBIAN_FRONTEND noninteractive
ENV LANG en_US.utf8
ENV MAKEFLAGS -j4

RUN mkdir /app
WORKDIR /app

RUN cp /etc/apt/sources.list /etc/apt/sources.list~
RUN sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list

# DEPENDENCIES
#===========================================
RUN apt-get update -y && \
    apt-get build-dep -y python3 && \
    apt-get install -y gcc make wget zlib1g-dev openssh-server \
                       build-essential gdb lcov pkg-config \
                       libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
                       libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
                       lzma lzma-dev tk-dev uuid-dev zlib1g-dev vim

# INSTALL PYTHON
#===========================================
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -zxf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --with-ensurepip=install --enable-shared && make && make install && \
    ldconfig && \
    ln -sf python3 /usr/local/bin/python
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install julia

# INSTALL JULIA
#====================================
RUN wget https://raw.githubusercontent.com/abelsiqueira/jill/main/jill.sh && \
    bash /app/jill.sh -y -v $JULIA_VERSION && \
    export PYTHON="python" && \
    julia -e 'using Pkg; Pkg.add("PyCall")' && \
    python -c 'import julia; julia.install()'

# CLEAN UP
#===========================================
RUN rm -rf /app/jill.sh \
    /opt/julias/*.tar.gz \
    /app/Python-3.8.5.tgz

WORKDIR /app/

COPY ./requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN julia -e 'using Pkg; Pkg.add([Pkg.PackageSpec(;name="EquivalentCircuits"),  \
                                  Pkg.PackageSpec(;name="CSV", version="0.10.2"),  \
                                  Pkg.PackageSpec(;name="DataFrames", version="1.3.2"),  \
                                  Pkg.PackageSpec(;name="JSON3", version="1.9.2"),  \
                                  Pkg.PackageSpec(;name="StringEncodings", version="0.3.5"),  \
                                  Pkg.PackageSpec(;name="PyCall", version="1.93.0") \
                                ])'

CMD ["/bin/bash"]
