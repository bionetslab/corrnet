# CorrNet

*A Python package for analysis of historical correspondence networks.*

## Requirements and installation

- The easiest way to install all dependencies of CorrNet is to install an Anaconda distribution: https://docs.anaconda.com/anaconda/install/

- Once Anaconda has been installed, download this repository into a directory `corrnet`. For this, either click on **Code &#8594; Download ZIP**, or download the directory via [git](https://git-scm.com/):

  ```sh
  git clone https://github.com/bionetslab/corrnet.git
  ```

  

## Usage

- Navigate to the `corrnet` directory:

  ```sh
  cd corrnet
  ```

- All dependencies of CorrNet are available in conda's default environment `base`. So you can simply run CorrNet in this enviroment, which you can activate as follows:

  ```sh
  conda activate
  ```

- Alternatively, you can generate and use a custom environment `corrnet` as follows:

  ```sh
  conda env create -f environment.yml
  conda activate corrnet
  (corrnet) python -m ipykernel install --user --name=corrnet
  ```

- Now, you can use conda as exemplified in the notebook `tutorial.ipynb`:

  ```sh
  jupyter notebook tutorial.ipynb
  ```

  
