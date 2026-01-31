# lopper

Interactive UI for pruning scikit-learn DecisionTreeRegressor models. Visualize your tree, collapse unwanted branches, and export a fully pruned model.

## Features

- **Interactive Visualization**: Drag-and-drop tree layout with auto-sized nodes
- **Click to Collapse/Expand**: Click nodes to collapse subtrees recursively; click again to expand direct children
- **Color-Coded Heatmap**: Nodes colored by any stat (white=average, red/blue=extremes)
- **Complete Pruning**: Exported trees have hidden nodes completely removed
- **Multiple Export Formats**: 
  - Pickle (`.pkl`) - Pruned tree ready to reload or use
  - Excel (`.xlsx`) - Node paths, splits, and statistics
  - PNG image - Visual snapshot of the tree

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Generate example tree:**
   ```bash
   uv run lopper-generate-example
   ```

3. **Run the app:**
   ```bash
   uv run lopper-app
   ```

The app opens automatically at `http://127.0.0.1:8050`.

## Using Your Own Tree

To use lopper with your own decision tree:

1. **Prepare your tree** with node statistics using the helper function:
   ```python
   from lopper.example_data import pickle_tree_with_stats
   import pandas as pd
   
   # Your fitted DecisionTreeRegressor
   tree = ...
   
   # Your training data
   df = pd.DataFrame(...)
   
   # Add statistics and save
   pickle_tree_with_stats(tree, df, "path/to/output.pkl")
   ```

2. **Update config.yml:**
   ```yaml
   tree_pickle_path: ./path/to/output.pkl
   heatmap_stat_key: mean  # or any stat key you want for coloring
   output_dir: ./outputs
   x_gap: 180  # horizontal spacing between leaves
   y_gap: 140  # vertical spacing between levels
   ```

3. **Run the app:**
   ```bash
   uv run lopper-app
   ```

## How to Use

1. **Navigate the tree**: Drag nodes to rearrange, zoom/pan to navigate
2. **Collapse branches**: Click a node to collapse its entire subtree
3. **Expand branches**: Click a collapsed node to expand its direct children only
4. **Bring nodes forward**: Click a node to bring it to the front (useful for overlapping nodes)
5. **Export**: Enter a name and click "Export" to save:
   - `name.pkl` - Pruned tree (reloadable in the app)
   - `name.xlsx` - Excel with node details
   - `name.png` - Screenshot of current view

## Configuration Reference

Edit `config.yml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `tree_pickle_path` | Path to pickled tree file | `./data/example_tree.pkl` |
| `heatmap_stat_key` | Stat key for node coloring | `mean` |
| `output_dir` | Export destination | `./outputs` |
| `x_gap` | Horizontal spacing (pixels) | `180` |
| `y_gap` | Vertical spacing (pixels) | `140` |

## Development

### Setup

Install dev dependencies (includes pre-commit hooks and ruff):
```bash
uv sync --extra dev
uv run pre-commit install
```

### Code Quality

Run linting and formatting:
```bash
uv run ruff check .
uv run ruff format .
```

Pre-commit hooks will automatically run ruff on staged files.

## Requirements

- Python 3.10-3.13
- `uv` package manager

## Project Structure

```
lopper/
├── lopper/
│   ├── app.py           # Flask + Cytoscape.js UI
│   └── example_data.py  # Example tree generator + helper function
├── data/                # Your tree pickle files
├── outputs/             # Exported files
├── config.yml          # Configuration
└── pyproject.toml      # Dependencies
```

## Notes

- The app doesn't auto-close after export - you can export multiple pruned versions from the same tree
- Exported trees are completely pruned - hidden nodes are physically removed, not just marked
- When you reload a pruned tree, only the remaining nodes will display
- Node statistics (`tree.node_stats`) must be a list of dicts, one per node

## License

This project uses MIT License.
