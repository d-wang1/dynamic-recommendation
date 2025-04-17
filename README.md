# dynamic-recommendation
Explore methods for dynamic recommendation systems for CSE543. Authored by David Wang, Shirley Li, Kathleen Weng, Donghong Cai, Xinhang Yuan.


## Setup:

During development, let's use a consistent set of packages.
1. Create a venv via `python -m venv venv`. I'm using `pip --version` == 22.0.4
2. Activate the venv via `source venv/bin/activate` on Mac/Linux and `.\venv\Scripts\activate` on Windows. 
   *Don't track the venv by adding it to the `.gitignore` file (via the following line: `venv`)*
3. Use `uv` for package handling. It should exist in the venv, but if it doesn't, `pip install uv`.
4. Run `uv init` 
5. To add new requirements / packages, run `uv add <package>`, e.g. `uv add pandas`
6. To install all the packages in pyproject.toml, run `uv sync`. This is the equivalent of using `pip install -r requirements.txt`.

(If pip is missing, use `python -m ensurepip --upgrade`)


1. Install pytorch for your specific CUDA version on https://pytorch.org/get-started/locally/. 
   (To check your CUDA version, run `nvidia-smi` on your terminal and look on the first row)
