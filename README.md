# COSA-Simulator

Opinion dynamics simulation in Python/Jupyter Notebook.

## Quick start

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook
   ```bash
   jupyter lab  # or: jupyter notebook
   ```

## Repository layout

- Jupyter notebooks:
  - `COSA - Simulator/Interface.ipynb`

## Reproducibility

- Environment: see `requirements.txt` (unversioned by default — pin versions after testing).
- To clear large cell outputs before committing, consider using [nbstripout](https://github.com/kynan/nbstripout):
  ```bash
  pip install nbstripout
  nbstripout --install
  ```

## How to cite

If you use this simulator in your research, please cite:
```text
@software{cosa-simulator,
  title = {COSA-Simulator},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-username/COSA-Simulator}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE).
