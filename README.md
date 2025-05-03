# dynamic-recommendation
Explore methods for dynamic recommendation systems for CSE543. Authored by David Wang, Shirley Li, Kathleen Weng, Donghong Cai, Xinhang Yuan.


## Setup:

   
1. Use `uv` for package handling. It should exist in the venv, but if it doesn't, `pip install uv`.
2. Run `uv init` 
3. Activate the venv via `source venv/bin/activate` on Mac/Linux and `.\venv\Scripts\activate` on Windows. 
   *Don't track the venv by adding it to the `.gitignore` file (via the following line: `venv`)*
4. To add new requirements / packages, run `uv add <package>`, e.g. `uv add pandas`
   *If you hand-installed any packages, run `uv lock`*
5. To install all the packages in pyproject.toml, run `uv sync --frozen`. This is the equivalent of using `pip install -r requirements.txt`.
   *This will fail if the pyproject.toml and uv.lock have diverged. If this happens, `uv lock -diff` can show the packages that bumped*

(If pip is missing, use `python -m ensurepip --upgrade`)

- Install pytorch for your specific CUDA version on https://pytorch.org/get-started/locally/. Add `uv` in front of the `pip` so the command looks like `uv pip install torch <etc...>`
   (To check your CUDA version, run `nvidia-smi` on your terminal and look on the first row)


# Training

- For logging, either set the comet API key via `export COMET_API_KEY="yourapikey"` or change the api key value in `config.json` (not recommended for final release)
- Activate venv via step 3 in setup
- `python train.py`


# Evaluation

- To change the checkpoint location, modify `app.ckpt_to_use` in `config.json`
- To get statistics such as MSRE and Precision@K, run `eval.py`
- To get sample user recommendations, run `test_outputs.py`