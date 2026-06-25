import sqlite3
import time
import csv
from pathlib import Path
from datetime import datetime
import logging

from qcn.engine.config import RESULTS_DIR

logger = logging.getLogger(__name__)

DB_PATH = RESULTS_DIR / "simulation_results.db"


def _get_connection() -> sqlite3.Connection:
    """Open a connection to the SQLite database, creating it if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT    NOT NULL,
                network_type TEXT    NOT NULL,
                sim_mode     TEXT    NOT NULL,
                nodes        INTEGER NOT NULL,
                radius       REAL    NOT NULL,
                density      REAL,
                mc_reps      INTEGER NOT NULL,
                seed         INTEGER,
                duration_s   REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id  INTEGER NOT NULL REFERENCES runs(id),
                metric  TEXT    NOT NULL,
                value   REAL    NOT NULL,
                step_n  INTEGER
            )
        """)
        conn.commit()
    logger.debug("Database initialised at %s", DB_PATH)


def save_run(result: dict, duration_s: float = 0.0) -> int:
    """
    Save a simulation result to the database.
    Returns the run_id of the saved run.
    """
    init_db()
    timestamp = datetime.utcnow().isoformat()
    mode = result.get("mode", "unknown")

    with _get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO runs
                (timestamp, network_type, sim_mode, nodes, radius, density, mc_reps, seed, duration_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            result.get("type"),
            mode,
            result.get("final_n"),
            result.get("final_radius"),
            result.get("density_val"),
            result.get("mc_reps", 1),
            result.get("seed"),
            duration_s,
        ))
        run_id = cursor.lastrowid

        # --- Save metrics depending on mode ---
        if mode == "distribution":
            dist_x = result.get("dist_x", [])
            dist_y = result.get("dist_y", [])
            for k, pk in zip(dist_x, dist_y):
                conn.execute(
                    "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                    (run_id, "degree_prob", float(pk), int(k))
                )
            # Summary metrics for distribution mode
            if result.get("mean_degree") is not None:
                conn.execute(
                    "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                    (run_id, "mean_degree", float(result["mean_degree"]), None)
                )
            if result.get("clustering") is not None:
                conn.execute(
                    "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                    (run_id, "clustering", float(result["clustering"]), None)
                )
            if result.get("ng_ratio") is not None:
                conn.execute(
                    "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                    (run_id, "ng_ratio", float(result["ng_ratio"]), None)
                )

        elif mode == "evolution":
            for n, l, d, c, k, ng in zip(
                result.get("x_nodes", []),
                result.get("y_path", []),
                result.get("y_diameter", []),
                result.get("y_clustering", []),
                result.get("y_mean_degree", []),
                result.get("y_ng_ratio", []),
            ):
                conn.execute(
                    "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                    (run_id, "path_length", float(l), int(n))
                )
                conn.execute(
                    "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                    (run_id, "diameter", float(d), int(n))
                )
                conn.execute(
                    "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                    (run_id, "clustering", float(c), int(n))
                )
                conn.execute(
                    "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                    (run_id, "mean_degree", float(k), int(n))
                )
                conn.execute(
                    "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                    (run_id, "ng_ratio", float(ng), int(n))
                )

            # Path length fit params
            fp = result.get("fit_params")
            if fp:
                fit_type = fp.get("type", "logarithmic")
                if fit_type == "logarithmic":
                    if fp.get("analytic"):
                        conn.execute(
                            "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                            (run_id, "fit_c_rho", float(fp["c_rho"]), None)
                        )
                        conn.execute(
                            "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                            (run_id, "fit_mean_degree", float(fp["mean_degree"]), None)
                        )
                    else:
                        conn.execute(
                            "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                            (run_id, "fit_a", float(fp["a"]), None)
                        )
                        conn.execute(
                            "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                            (run_id, "fit_b", float(fp["b"]), None)
                        )
                elif fit_type == "powerlaw":
                    conn.execute(
                        "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                        (run_id, "fit_b", float(fp["b"]), None)
                    )
                    conn.execute(
                        "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                        (run_id, "fit_alpha", float(fp["alpha"]), None)
                    )

            # Diameter fit params
            fp_diam = result.get("fit_params_diam")
            if fp_diam:
                fit_type = fp_diam.get("type", "logarithmic")
                if fit_type == "logarithmic":
                    if fp.get("analytic"):
                        conn.execute(
                            "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                            (run_id, "fit_c_rho", float(fp["c_rho"]), None)
                        )
                        conn.execute(
                            "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                            (run_id, "fit_mean_degree", float(fp["mean_degree"]), None)
                        )
                    else:
                        conn.execute(
                            "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                            (run_id, "fit_a", float(fp["a"]), None)
                        )
                        conn.execute(
                            "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                            (run_id, "fit_b", float(fp["b"]), None)
                        )
                elif fit_type == "powerlaw":
                    conn.execute(
                        "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                        (run_id, "diam_fit_b", float(fp_diam["b"]), None)
                    )
                    conn.execute(
                        "INSERT INTO metrics (run_id, metric, value, step_n) VALUES (?, ?, ?, ?)",
                        (run_id, "diam_fit_alpha", float(fp_diam["alpha"]), None)
                    )

        conn.commit()
    logger.info("Run saved: id=%d type=%s mode=%s", run_id, result.get("type"), mode)
    return run_id

def get_runs(network_type: str = None, sim_mode: str = None) -> list[dict]:
    """
    Retrieve all runs, optionally filtered by network_type and/or sim_mode.
    Returns a list of dicts.
    """
    init_db()
    query = "SELECT * FROM runs WHERE 1=1"
    params = []
    if network_type:
        query += " AND network_type = ?"
        params.append(network_type)
    if sim_mode:
        query += " AND sim_mode = ?"
        params.append(sim_mode)
    query += " ORDER BY timestamp DESC"

    with _get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def get_metrics(run_id: int, metric: str = None) -> list[dict]:
    """
    Retrieve metrics for a given run_id, optionally filtered by metric name.
    """
    init_db()
    query = "SELECT * FROM metrics WHERE run_id = ?"
    params = [run_id]
    if metric:
        query += " AND metric = ?"
        params.append(metric)
    # Order by step_n with NULLs last so summary metrics appear after per-step data
    query += " ORDER BY step_n IS NULL, step_n"

    with _get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]

def export_csv(run_id: int, output_path: Path = None) -> Path:
    """
    Export all metrics for a run to a CSV file.
    Returns the path to the written file.
    """
    init_db()
    runs = get_runs()
    run = next((r for r in runs if r["id"] == run_id), None)
    if not run:
        raise ValueError(f"Run {run_id} not found")

    if output_path is None:
        output_path = RESULTS_DIR / f"run_{run_id}_{run['network_type']}_{run['sim_mode'].replace(' ', '_')}.csv"

    metrics = get_metrics(run_id)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "run_id", "metric", "value", "step_n"])
        writer.writeheader()
        writer.writerows(metrics)

    logger.info("Exported run %d to %s", run_id, output_path)
    return output_path