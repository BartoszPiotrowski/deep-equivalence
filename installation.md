# Installation

To run the stuff Python 3 and Tensorflow is needed. But for example Python 3.5.2
and Tensorflow 1.6 doesn't work properly with this code, as opposed to Python
3.6.6 with Tensorflow 1.10. The easiest way how to satisfy these requirements is
to install Anaconda -- Python distribution which simplify package management. To
install it follow these instructions:

1. Download the Anaconda installer:
```
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
```
2. Run the installer. (Accept the license terms, choose the installation lacation,
   prepend the Anaconda install location to your `PATH` in `.bashrc`.)
```
bash Anaconda3-5.2.0-Linux-x86_64.sh
```
3. Create a new environment (with some arbitrary name like `deep-learning`) and
   install `tensorflow` there:
```
conda create --name deep-learning tensorflow
```
4. Activate the created environment:
```
conda activate deep-learning
```

And now you can run experiments, as described in `README.md`.
