# syntax = docker/dockerfile:1.2
ARG UBUNTU_RELEASE=20.04
ARG SOURCE_DIR=/home/app/

FROM ubuntu:$UBUNTU_RELEASE
ARG SOURCE_DIR
ENV SOURCE_DIR $SOURCE_DIR
ENV PATH $PATH:$SOURCE_DIR
RUN mkdir -p $SOURCE_DIR
WORKDIR $SOURCE_DIR
RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --shell /bin/bash --create-home app \
 # install pkgs
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    # you might need build-essential
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    # other pkgs...
 && rm -rf /var/lib/apt/lists/*
# make some useful symlinks
RUN cd /usr/local/bin \
 && ln -s /usr/bin/python3 python \
 && ln -s /usr/bin/python3-config python-config \
 && ln -s /usr/local/bin/glpsol glpk
COPY --chown=app:app ./requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
COPY --chown=app:app ./*.sh ./
COPY --chown=app:app ./src/ ./src/

USER root

# Install wget
RUN apt-get update -y && apt-get install -y \
	wget \
	build-essential \
	--no-install-recommends \
	&& rm -rf /var/lib/apt/lists/*

# Install glpk from http
# instructions and documentation for glpk: http://www.gnu.org/software/glpk/
WORKDIR /user/local/
RUN wget http://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz \
	&& tar -zxvf glpk-5.0.tar.gz

## Verify package contents
# RUN wget http://ftp.gnu.org/gnu/glpk/glpk-5.0.tar.gz.sig \
#	&& gpg --verify glpk-5.0.tar.gz.sig
#	#&& gpg --keyserver keys.gnupg.net --recv-keys 5981E818

WORKDIR /user/local/glpk-5.0
RUN ./configure \
	&& make \
	&& make check \
	&& make install \
	&& make distclean \
	&& ldconfig \
# Cleanup
	&& rm -rf /user/local/glpk-5.0.tar.gz \
	&& apt-get clean

#create a glpk user
ENV HOME /home/user
RUN useradd --create-home --home-dir $HOME user \
    && chmod -R u+rwx $HOME \
    && chown -R user:user $HOME



WORKDIR $SOURCE_DIR

USER app
CMD ["/bin/bash"]