# Data_Science_Python

![Logo](./assets/Logo.png)

A comprehensive collection of Python notebooks and source code for scientific computing, data analysis, and machine learning. Started in 2018 and maintained as a modern learning resource.

## Project Structure

### `notebooks/` (Interactive Learning)
- **`01_basics`**: Fundamentals, NumPy, Pandas, SQLite, Datashader.
- **`02_math_stats`**: Probability, Statistics, and Simulations.
- **`03_deep_learning`**: TensorFlow (Keras, TF2.x) and PyTorch tutorials.
- **`04_nlp`**: NLP, Word Embeddings, and Transformers.
- **`05_cv`**: Computer Vision & Segmentation.
- **`06_tabular_ml`**: Scikit-Learn, Gradient Boosting, and AutoML.
- **`07_big_data`**: PySpark and distributed computing.
- **`08_algorithms`**: Sorting, Searching, and Algorithm Case Studies.
- **`09_geospatial`**: GIS and Spatial Data Analysis.
- **`10_visualization`**: Bokeh, Folium, and TensorBoard.

### `src/` & `scripts/` (Source & Infrastructure)
- **`src/`**: Modular implementations of algorithms, geospatial tools, and web services (FastAPI/Tornado).
- **`scripts/`**: Development utilities and linter configurations.
- **`assets/`**: Shared media and documentation assets.

## Setup & Development

### Requirements
- **Python 3.12+** (Recommended)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download/)

### Environment Setup
```bash
# Create and activate environment
conda create --name ds_python python=3.12
conda activate ds_python

# Install core stack
conda install numpy pandas matplotlib scikit-learn ruff pre-commit
```

### Standards
This project uses **pre-commit** hooks to ensure quality:
- **Ruff**: Fast linting and formatting (replaces Black/Flake8).
- **nb-clean**: Automatically clears notebook outputs and metadata.

## License
Distributed under the [MIT License](MIT License.txt).
