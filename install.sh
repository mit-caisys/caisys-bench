#!/bin/bash
set -xe

CONDA_BASE_DIR="/mnt"
CONDA_DIR="${CONDA_BASE_DIR}/miniconda3"
CONDA_BIN="${CONDA_DIR}/condabin"
CONDA_ACTIVATE="${CONDA_DIR}/bin/activate"
CONDA=${CONDA_BIN}/conda
CONDA_ENV=".comp-ai-bench"

install_conda() {
  if [ -d ${CONDA_DIR} ]; then
    echo "miniconda exists"
    return
  fi

  pushd .
  cd ${CONDA_BASE_DIR}
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -u -p ${CONDA_DIR}
  rm -rf miniconda.sh
  ${CONDA} init bash
  popd
}

create_conda_env() {
  pushd .
  ${CONDA} init
  source ${HOME}/.bashrc
  ${CONDA} create -n ${CONDA_ENV} -y python=3.12.7
  activate_conda_env
  ${CONDA} install -y -c conda-forge git-lfs
  git lfs install
  python -m pip install -r requirements-core.txt
  python -m pip install -r requirements-analysis.txt
  python -m pip install -r requirements-browser.txt
  popd
}

activate_conda_env() {
  ${CONDA} init
  source ${HOME}/.bashrc
  source ${CONDA_ACTIVATE} ${CONDA_ENV}
  which python
}

install_all() {
  echo "Running complete installation."

  install_conda
  create_conda_env
  activate_conda_env

  echo "Installation complete."
}
