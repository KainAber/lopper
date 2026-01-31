from __future__ import annotations

import base64
import copy
import datetime as dt
import threading
import webbrowser
from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from flask import Flask, jsonify, render_template_string, request
from sklearn.tree import DecisionTreeRegressor

APP_TITLE = "Lopper"


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    return {
        "tree_pickle_path": Path(config.get("tree_pickle_path", "")),
        "heatmap_stat_key": str(config.get("heatmap_stat_key", "mean")),
        "output_dir": Path(config.get("output_dir", "./outputs")),
        "x_gap": int(config.get("x_gap", 180)),
        "y_gap": int(config.get("y_gap", 140)),
    }


def validate_tree(model: DecisionTreeRegressor) -> None:
    if not hasattr(model, "tree_"):
        raise ValueError("Loaded object is not a fitted DecisionTreeRegressor.")
    if not hasattr(model, "node_stats"):
        raise ValueError("Tree is missing node_stats. Generate the pickle with node stats included.")


def blend_color(base: tuple[int, int, int], target: tuple[int, int, int], amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    blended = tuple(int(round(b + (t - b) * amount)) for b, t in zip(base, target, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*blended)


def color_for_values(values: Iterable[float]) -> list[str]:
    values = list(values)
    if not values:
        return []
    mean_val = float(np.mean(values))
    max_dev = max(abs(v - mean_val) for v in values) or 1.0

    blue = (78, 121, 167)
    red = (225, 87, 89)
    white = (255, 255, 255)

    colors = []
    for value in values:
        ratio = (value - mean_val) / max_dev
        if ratio >= 0:
            colors.append(blend_color(white, red, ratio))
        else:
            colors.append(blend_color(white, blue, abs(ratio)))
    return colors


def format_stats(stats: dict) -> list[str]:
    lines = []
    for key in sorted(stats.keys()):
        value = stats[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4g}")
        else:
            lines.append(f"{key}: {value}")
    return lines


def compute_positions(
    children_left: np.ndarray, children_right: np.ndarray, x_gap: int = 180, y_gap: int = 140
) -> dict[int, dict[str, float]]:
    positions: dict[int, dict[str, float]] = {}
    x_pos = 0

    def dfs(node_id: int, depth: int) -> None:
        nonlocal x_pos
        left = children_left[node_id]
        right = children_right[node_id]

        if left == -1 and right == -1:
            positions[node_id] = {"x": x_pos * x_gap, "y": depth * y_gap}
            x_pos += 1
            return

        child_positions = []
        if left != -1:
            dfs(left, depth + 1)
            child_positions.append(positions[left]["x"])
        if right != -1:
            dfs(right, depth + 1)
            child_positions.append(positions[right]["x"])

        positions[node_id] = {"x": float(np.mean(child_positions)), "y": depth * y_gap}

    dfs(0, 0)
    return positions


def build_tree_maps(tree) -> tuple[dict[int, list[int]], dict[int, int]]:
    children_left = tree.children_left
    children_right = tree.children_right
    children_map: dict[int, list[int]] = {}
    parent_map: dict[int, int] = {}

    for node_id in range(tree.node_count):
        left = children_left[node_id]
        right = children_right[node_id]
        children = []
        if left != -1:
            children.append(int(left))
            parent_map[int(left)] = node_id
        if right != -1:
            children.append(int(right))
            parent_map[int(right)] = node_id
        children_map[node_id] = children

    return children_map, parent_map


def get_hidden_nodes(children_map: dict[int, list[int]], collapsed_nodes: set[int]) -> set[int]:
    hidden: set[int] = set()
    for node_id in collapsed_nodes:
        stack = list(children_map.get(node_id, []))
        while stack:
            current = stack.pop()
            if current in hidden:
                continue
            hidden.add(current)
            stack.extend(children_map.get(current, []))
    return hidden


def collapsed_roots(collapsed_nodes: set[int], parent_map: dict[int, int]) -> list[int]:
    roots = []
    for node_id in collapsed_nodes:
        current = parent_map.get(node_id)
        while current is not None:
            if current in collapsed_nodes:
                break
            current = parent_map.get(current)
        else:
            roots.append(node_id)
    return roots


def prune_tree(model: DecisionTreeRegressor, collapsed_nodes: set[int]) -> DecisionTreeRegressor:
    """
    Completely prune the tree by physically removing hidden nodes.
    Uses sklearn's internal _resize method to rebuild with only reachable nodes.
    """
    if not collapsed_nodes:
        # No nodes to prune, return a copy
        return copy.deepcopy(model)

    old_tree = model.tree_
    children_map, parent_map = build_tree_maps(old_tree)

    # Mark collapsed nodes as leaves first
    pruned = copy.deepcopy(model)
    tree = pruned.tree_

    for node_id in collapsed_roots(collapsed_nodes, parent_map):
        if not children_map[node_id]:
            continue
        tree.children_left[node_id] = -1
        tree.children_right[node_id] = -1
        tree.feature[node_id] = -2
        tree.threshold[node_id] = -2.0

    # Get all reachable nodes after collapsing
    reachable = reachable_nodes(tree)
    hidden = set(range(tree.node_count)) - set(reachable)

    if not hidden:
        # No nodes were actually removed
        return pruned

    # Create mapping from old node IDs to new node IDs
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(reachable)}
    n_new_nodes = len(reachable)

    # Build new arrays with only reachable nodes
    new_children_left = np.full(n_new_nodes, -1, dtype=np.intp)
    new_children_right = np.full(n_new_nodes, -1, dtype=np.intp)
    new_feature = np.full(n_new_nodes, -2, dtype=np.intp)
    new_threshold = np.zeros(n_new_nodes, dtype=np.float64)
    new_value = np.zeros((n_new_nodes, tree.n_outputs, tree.value.shape[2]), dtype=np.float64)
    new_impurity = np.zeros(n_new_nodes, dtype=np.float64)
    new_n_node_samples = np.zeros(n_new_nodes, dtype=np.intp)
    new_weighted_n_node_samples = np.zeros(n_new_nodes, dtype=np.float64)

    # Copy data for reachable nodes
    for new_id, old_id in enumerate(reachable):
        new_feature[new_id] = tree.feature[old_id]
        new_threshold[new_id] = tree.threshold[old_id]
        new_value[new_id] = tree.value[old_id]
        new_impurity[new_id] = tree.impurity[old_id]
        new_n_node_samples[new_id] = tree.n_node_samples[old_id]
        new_weighted_n_node_samples[new_id] = tree.weighted_n_node_samples[old_id]

        # Update child pointers
        left_child = tree.children_left[old_id]
        right_child = tree.children_right[old_id]

        if left_child != -1 and left_child in old_to_new:
            new_children_left[new_id] = old_to_new[left_child]
        if right_child != -1 and right_child in old_to_new:
            new_children_right[new_id] = old_to_new[right_child]

    # Directly modify the internal arrays (they are numpy arrays, which are writable)
    tree.children_left[:] = -1
    tree.children_right[:] = -1
    tree.children_left[:n_new_nodes] = new_children_left
    tree.children_right[:n_new_nodes] = new_children_right
    tree.feature[:n_new_nodes] = new_feature
    tree.threshold[:n_new_nodes] = new_threshold
    tree.value[:n_new_nodes] = new_value
    tree.impurity[:n_new_nodes] = new_impurity
    tree.n_node_samples[:n_new_nodes] = new_n_node_samples
    tree.weighted_n_node_samples[:n_new_nodes] = new_weighted_n_node_samples

    # Update node count
    tree.node_count = n_new_nodes

    # Recalculate max_depth
    def calculate_depth(node_id, depth=0):
        if node_id == -1:
            return depth - 1
        left_depth = calculate_depth(tree.children_left[node_id], depth + 1)
        right_depth = calculate_depth(tree.children_right[node_id], depth + 1)
        return max(depth, left_depth, right_depth)

    tree.max_depth = calculate_depth(0)

    # Update node_stats if present
    if hasattr(pruned, "node_stats"):
        new_stats = [model.node_stats[old_id] for old_id in reachable]
        pruned.node_stats = new_stats

    return pruned


def build_paths(tree, feature_names: list[str]) -> dict[int, str]:
    children_left = tree.children_left
    children_right = tree.children_right
    paths: dict[int, str] = {}

    def dfs(node_id: int, path_parts: list[str]) -> None:
        paths[node_id] = "\n AND ".join(path_parts) if path_parts else "ROOT"
        left = children_left[node_id]
        right = children_right[node_id]
        if left == -1 and right == -1:
            return
        feature_name = feature_names[tree.feature[node_id]]
        threshold = tree.threshold[node_id]
        if left != -1:
            dfs(int(left), path_parts + [f"{feature_name} <= {threshold:.4g}"])
        if right != -1:
            dfs(int(right), path_parts + [f"{feature_name} > {threshold:.4g}"])

    dfs(0, [])
    return paths


def reachable_nodes(tree) -> list[int]:
    children_left = tree.children_left
    children_right = tree.children_right
    visited = []
    stack = [0]
    seen = set()
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        visited.append(node_id)
        left = children_left[node_id]
        right = children_right[node_id]
        if left != -1:
            stack.append(int(left))
        if right != -1:
            stack.append(int(right))
    return visited


def save_excel(tree_model: DecisionTreeRegressor, output_path: Path) -> None:
    tree = tree_model.tree_
    feature_names = list(getattr(tree_model, "feature_names_in_", []))
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(tree.n_features)]

    paths = build_paths(tree, feature_names)
    rows = []
    for node_id in reachable_nodes(tree):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        is_leaf = left == -1 and right == -1

        row = {"node_id": node_id, "path": paths[node_id]}
        if is_leaf:
            row["split_feature"] = None
            row["split_threshold"] = None
        else:
            row["split_feature"] = feature_names[tree.feature[node_id]]
            row["split_threshold"] = tree.threshold[node_id]

        if hasattr(tree_model, "node_stats") and node_id < len(tree_model.node_stats):
            stats = tree_model.node_stats[node_id]
            # Only add stats if they exist (skip hidden nodes with empty dicts)
            if stats:
                for key, val in stats.items():
                    row[f"stat_{key}"] = val

        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)


def safe_export_name(name: str) -> str:
    name = name.strip() or f"export_{dt.datetime.now():%Y%m%d_%H%M%S}"
    name = "".join(c if c.isalnum() or c in "_- " else "_" for c in name)
    return name


def save_tree_pickle(model: DecisionTreeRegressor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def build_elements(model: DecisionTreeRegressor, heatmap_key: str, positions: dict[int, dict[str, float]]) -> dict:
    tree = model.tree_
    children_left = tree.children_left
    children_right = tree.children_right
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(tree.n_features)]

    heatmap_values = []
    for node_id in range(tree.node_count):
        if hasattr(model, "node_stats") and node_id < len(model.node_stats):
            stats = model.node_stats[node_id]
            heatmap_values.append(stats.get(heatmap_key, 0.0))
        else:
            heatmap_values.append(0.0)

    colors = color_for_values(heatmap_values)

    nodes = []
    edges = []

    for node_id in range(tree.node_count):
        left = children_left[node_id]
        right = children_right[node_id]
        is_leaf = left == -1 and right == -1

        if is_leaf:
            label = f"Node {node_id}"
        else:
            feat_name = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            label = f"Node {node_id}\n\n{feat_name} <= {threshold:.4g}"

        if hasattr(model, "node_stats") and node_id < len(model.node_stats):
            stats = model.node_stats[node_id]
            stat_lines = format_stats(stats)
            label += "\n\n" + "\n".join(stat_lines)

        pos = positions.get(node_id, {"x": 0, "y": 0})
        nodes.append(
            {
                "data": {
                    "id": str(node_id),
                    "label": label,
                    "color": colors[node_id],
                },
                "position": {"x": pos["x"], "y": pos["y"]},
            }
        )

        if left != -1:
            edges.append({"data": {"source": str(node_id), "target": str(left)}})
        if right != -1:
            edges.append({"data": {"source": str(node_id), "target": str(right)}})

    return {"nodes": nodes, "edges": edges}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 16px;
        }
        h2 { margin-top: 0; }
        #cy {
            width: 100%;
            height: 80vh;
            border: 1px solid #ccc;
            background-color: #fafafa;
        }
        .controls {
            margin-bottom: 12px;
        }
        .controls input, .controls button {
            margin-right: 8px;
            padding: 6px 12px;
        }
        #status {
            margin-left: 12px;
            color: #666;
        }
        #debug {
            font-size: 11px;
            color: #666;
            margin-bottom: 8px;
            display: none;
        }
    </style>
</head>
<body>
    <h2>{{ title }}</h2>
    <p>Click a node to collapse its subtree. Click again to expand to direct children.</p>
    <div class="controls">
        <input type="text" id="export-name" placeholder="export name" value="{{ default_export_name }}" />
        <button id="export-btn">Export</button>
        <span id="status"></span>
    </div>
    <div id="cy"></div>

    <script>
        let collapsedNodes = new Set();
        const childrenMap = {{ children_map | tojson }};

        const cy = cytoscape({
            container: document.getElementById('cy'),
            elements: {{ elements | tojson }},
            style: [
                {
                    selector: 'node',
                    style: {
                        'label': 'data(label)',
                        'text-wrap': 'wrap',
                        'text-max-width': '200px',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'shape': 'round-rectangle',
                        'background-color': 'data(color)',
                        'border-width': 1,
                        'border-color': '#333',
                        'font-size': '10px',
                        'font-weight': 'bold',
                        'width': 'label',
                        'height': 'label',
                        'padding': '10px',
                    }
                },
                {
                    selector: '.hidden',
                    style: {
                        'display': 'none'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'width': 1,
                        'line-color': '#999',
                        'target-arrow-color': '#999'
                    }
                }
            ],
            layout: { name: 'preset' },
            minZoom: 0.2,
            maxZoom: 2.0,
            userPanningEnabled: true,
            userZoomingEnabled: true,
            boxSelectionEnabled: false,
        });

        function getHiddenNodes(collapsed) {
            const hidden = new Set();
            for (const nodeId of collapsed) {
                const stack = [...(childrenMap[nodeId] || [])];
                while (stack.length > 0) {
                    const current = stack.pop();
                    if (hidden.has(current)) continue;
                    hidden.add(current);
                    stack.push(...(childrenMap[current] || []));
                }
            }
            return hidden;
        }

        function updateVisibility() {
            const hidden = getHiddenNodes(collapsedNodes);
            cy.nodes().forEach(node => {
                const nodeId = parseInt(node.id());
                if (hidden.has(nodeId)) {
                    node.addClass('hidden');
                } else {
                    node.removeClass('hidden');
                }
            });
            cy.edges().forEach(edge => {
                const source = parseInt(edge.data('source'));
                const target = parseInt(edge.data('target'));
                if (hidden.has(source) || hidden.has(target)) {
                    edge.addClass('hidden');
                } else {
                    edge.removeClass('hidden');
                }
            });
        }

        cy.on('tap', 'node', function(evt) {
            const node = evt.target;
            const nodeId = parseInt(node.id());
            console.log('[CLICK] Node', nodeId);

            // Bring clicked node to front
            node.style('z-index', 999);

            // Reset other nodes to default z-index
            cy.nodes().not(node).style('z-index', 1);

            if (collapsedNodes.has(nodeId)) {
                collapsedNodes.delete(nodeId);
                // Expand: add direct children to collapsed set
                const children = childrenMap[nodeId] || [];
                children.forEach(child => collapsedNodes.add(child));
            } else {
                collapsedNodes.add(nodeId);
            }

            updateVisibility();
        });

        document.getElementById('export-btn').addEventListener('click', async function() {
            const name = document.getElementById('export-name').value.trim();
            const status = document.getElementById('status');
            status.innerText = 'Exporting...';

            try {
                const png = cy.png({ output: 'base64', bg: '#fafafa', full: true });
                const response = await fetch('/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: name,
                        collapsed: Array.from(collapsedNodes),
                        image: png
                    })
                });
                const result = await response.json();
                status.innerText = result.message || 'Export complete!';
            } catch (err) {
                status.innerText = 'Export failed: ' + err.message;
            }
        });
    </script>
</body>
</html>
"""


def build_app(
    model: DecisionTreeRegressor, heatmap_key: str, output_dir: Path, tree_name: str, x_gap: int, y_gap: int
) -> Flask:
    app = Flask(__name__)
    app.config["OUTPUT_DIR"] = output_dir

    tree = model.tree_
    positions = compute_positions(tree.children_left, tree.children_right, x_gap, y_gap)
    elements = build_elements(model, heatmap_key, positions)
    children_map, _ = build_tree_maps(tree)

    @app.route("/")
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            title=APP_TITLE,
            elements=elements,
            children_map={str(k): v for k, v in children_map.items()},
            default_export_name=f"{tree_name}_pruned",
        )

    @app.route("/export", methods=["POST"])
    def export():
        data = request.json
        name = safe_export_name(data.get("name", ""))
        collapsed_list = data.get("collapsed", [])
        collapsed_nodes = {int(n) for n in collapsed_list}
        image_b64 = data.get("image", "")

        pruned = prune_tree(model, collapsed_nodes)

        output_dir = Path(app.config["OUTPUT_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)

        pickle_path = output_dir / f"{name}.pkl"
        excel_path = output_dir / f"{name}.xlsx"
        image_path = output_dir / f"{name}.png"

        save_tree_pickle(pruned, pickle_path)
        save_excel(pruned, excel_path)

        if image_b64:
            header, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
            with image_path.open("wb") as f:
                f.write(base64.b64decode(encoded))

        return jsonify({"message": f"Saved {pickle_path.name}, {excel_path.name}, {image_path.name}."})

    return app


def main(config_path: str | Path = None) -> None:
    config_path = config_path or Path(__file__).resolve().parents[1] / "config.yml"
    config = load_config(config_path)

    tree_path = config["tree_pickle_path"]
    if not tree_path.is_absolute():
        tree_path = (Path(__file__).resolve().parents[1] / tree_path).resolve()
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree pickle not found: {tree_path}")

    model = joblib.load(tree_path)
    validate_tree(model)

    # Extract tree name from the pickle file path
    tree_name = tree_path.stem  # Gets filename without extension

    heatmap_key = config["heatmap_stat_key"]
    output_dir = (Path(__file__).resolve().parents[1] / config["output_dir"]).resolve()
    x_gap = config["x_gap"]
    y_gap = config["y_gap"]
    app = build_app(model, heatmap_key, output_dir, tree_name, x_gap, y_gap)

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050")

    threading.Timer(1.0, open_browser).start()
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
