# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.231.4/containers/python-3/.devcontainer/base.Dockerfile

# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
# ARG VARIANT="3.10-bullseye"
# FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}
FROM thomasdwright/lbuc-jupyter:1.0

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
USER root
#RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# For some inane reason we need to do this twice
COPY software/requirements.txt /tmp/pip-tmp/
RUN sage -pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt; sage -pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends git

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1
ENV SAGE_REPO_AUTHENTICATED=ssh://git@trac.sagemath.org:2222/sage.git SAGE_VENV=/home/sage/sage/local SAGE_DOC=/home/sage/sage/local/share/doc/sage SAGE_LOCAL=/home/sage/sage/local IPYTHONDIR=/home/sage/.sage//ipython-5.0.0 SAGE_DOC_SRC=/home/sage/sage/src/doc SAGE_SRC=/home/sage/sage/src SINGULARPATH=/home/sage/sage/local/share/singular PYTHON_EGG_CACHE=/home/sage/.sage/.python-eggs SAGE_ROOT=/home/sage/sage SAGE_LOGS=/home/sage/sage/logs/pkgs SAGE_SPKG_INST=/home/sage/sage/local/var/lib/sage/installed SAGE_NUM_THREADS=1 SAGE_DISTFILES=/home/sage/sage/upstream SAGE_REPO_ANONYMOUS=https://gitlab.com/sagemath/dev/tracmirror.git LIBRARY_PATH=/home/sage/sage/local/lib SAGE_NUM_THREADS_PARALLEL=8 SAGE_ORIG_PATH_SET=True SAGE_STARTUP_FILE=/home/sage/.sage//init.sage SAGE_SHARE=/home/sage/sage/local/share SAGE_ENV_CONFIG_SOURCED=1 PATH=/home/sage:/home/sage/sage/build/bin:/home/sage/sage/src/bin:/home/sage/sage/local/bin:/home/sage/sage/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin PYTHONUSERBASE=/home/sage/.sage//local SAGE_ORIG_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin SAGE_ENV_SOURCED=4 DOT_SAGE=/home/sage/.sage/ CPATH=/home/sage/sage/local/include SAGE_DOC_MATHJAX=True
ENTRYPOINT ["/usr/local/bin/sage-entrypoint"]