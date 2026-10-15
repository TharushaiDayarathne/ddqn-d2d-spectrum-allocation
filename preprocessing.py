# preprocessing.py
# This builds:
    # Matching
    # Gain matrices
    # Power vectors
    # Snapshot dictionary
import pandas as pd
import numpy as np

def build_snapshots(file_path):

    alpha = 3
    P_D2D = 0.1
    P_BS = 1.0
    noise_power = 4e-15

    all_snapshots_data = {}

    xls = pd.ExcelFile(file_path)

    for sheet_name in xls.sheet_names:

        df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

        tx_columns = [col for col in df.columns if col.startswith("Tx")]
        bs_columns = [col for col in df.columns if col.startswith("BS")]

        tx_matrix = df[tx_columns].values
        bs_matrix = df[bs_columns].values

        n = len(df)

        # ---------- Matching (Priority Greedy) ----------
        gaps = []

        for i in range(n):
            distances = list(tx_matrix[i]) + list(bs_matrix[i])
            sorted_vals = sorted(distances)
            gap = sorted_vals[1] - sorted_vals[0]
            gaps.append((i, gap))

        gaps.sort(key=lambda x: -x[1])

        used_tx = set()
        matching = {}

        for i, _ in gaps:
            combined = []

            for j in range(len(tx_columns)):
                combined.append(("Tx", j, tx_matrix[i][j]))

            for j in range(len(bs_columns)):
                combined.append(("BS", j, bs_matrix[i][j]))

            combined.sort(key=lambda x: (x[2], 0 if x[0] == "Tx" else 1))

            for node_type, idx, dist in combined:
                if node_type == "Tx":
                    if idx not in used_tx:
                        used_tx.add(idx)
                        matching[i] = ("Tx", idx)
                        break
                else:
                    matching[i] = ("BS", idx)
                    break

        # ---------- Gain Matrix ----------
        full_distance_matrix = np.hstack((tx_matrix, bs_matrix))
        full_distance_matrix[full_distance_matrix == 0] = 1e-6

        fading = np.random.exponential(1, size=full_distance_matrix.shape)
        gain_matrix = fading / (full_distance_matrix ** alpha)

        serving_tx_indices = []
        power_vector = []

        for i in range(n):
            node_type, idx = matching[i]

            if node_type == "Tx":
                serving_tx_indices.append(idx)
                power_vector.append(P_D2D)
            else:
                serving_tx_indices.append(len(tx_columns) + idx)
                power_vector.append(P_BS)

        all_snapshots_data[sheet_name] = {
            "gain_matrix": gain_matrix,
            "serving_tx": np.array(serving_tx_indices),
            "power": np.array(power_vector),
            "noise": noise_power
        }

    return all_snapshots_data
