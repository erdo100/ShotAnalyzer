import os
import pandas as pd
import numpy as np

def read_eval_bbt_v02(directory):
    results = []
    results.append(['Filename', 'Offset', 'B2 Pos X', 'B2 Pos Y', 'Hit Thickness', 'B1_V', 'B1_hit',
                    'B1 pos1 X', 'B1 Pos1 Y', 'B1 pos2 X', 'B1 Pos2 Y', 'B1 pos3 X', 'B1 Pos3 Y',
                    'B1 pos4 X', 'B1 Pos4 Y', 'B1 Angle 1', 'B1 Angle 2', 'B2 Angle'])

    for file in os.listdir(directory):
        if file.endswith(".bbt"):
            filepath = os.path.join(directory, file)
            data = pd.read_csv(filepath, header=None).to_numpy()

            # Example calculations (replace with actual logic from MATLAB file)
            offset = np.mean(data[:, 0])
            b2_pos_x = np.mean(data[:, 1])
            b2_pos_y = np.mean(data[:, 2])
            hit_thickness = np.std(data[:, 3])
            b1_v = np.max(data[:, 4])
            b1_hit = np.sum(data[:, 5])

            # Placeholder for B1 positions and angles
            b1_positions = data[:4, 1:3].flatten()
            b1_angles = [np.arctan2(data[i, 2], data[i, 1]) for i in range(2)]

            # Append results
            results.append([file, offset, b2_pos_x, b2_pos_y, hit_thickness, b1_v, b1_hit] +
                           b1_positions.tolist() + b1_angles)

    # Save results to an Excel file
    output_path = os.path.join(directory, "BBT_Evaluation_Results.xlsx")
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False, header=False)
    print(f"Results saved to {output_path}")