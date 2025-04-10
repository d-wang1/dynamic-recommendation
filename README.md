# dynamic-recommendation
Explore methods for dynamic recommendation systems for CSE543. Authored by David Wang, Shirley Li, Kathleen Weng, Donghong Cai, Xinhang Yuan.


## Setup:

During development, let's use a consistent set of packages.
1. Create a venv via `python -m venv venv`. I'm using `pip --version` == 22.0.4
2. Activate the venv via `source venv/bin/activate` on Mac/Linux and `.\venv\Scripts\activate` on Windows. 
   *Don't track the venv by adding it to the `.gitignore` file (via the following line: `venv`)*
3. Whenever you install a new package, remember to create a corresponding requirements file via `pip freeze > requirements.txt`
4. Install from the requirements file using `pip install -r requirements.txt`
(If pip is missing, use `python -m ensurepip --upgrade`)


5. Install pytorch for your specific CUDA version on https://pytorch.org/get-started/locally/. 
   (To check your CUDA version, run `nvidia-smi` on your terminal and look on the first row)
