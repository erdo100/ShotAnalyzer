import os
import pandas as pd
import numpy as np

def read_eval_bbt(directory):
    results = [
        ['Filename', 'Offset', 'B2 Pos X', 'B2 Pos Y', 'Hit Thickness', 'B1_V', 'B1_hit',
         'B1 pos1 X', 'B1 Pos1 Y', 'B1 pos2 X', 'B1 Pos2 Y', 'B1 pos3 X', 'B1 Pos3 Y',
         'B1 pos4 X', 'B1 Pos4 Y', 'B1 Angle 1', 'B1 Angle 2', 'B2 Angle']
    ]

    for file in os.listdir(directory):
        if file.endswith(".bbt"):
            filepath = os.path.join(directory, file)
            data = pd.read_csv(filepath, header=None).to_numpy()

            # Find when B2 is moving
            i1 = np.argmax(np.abs(np.diff(data[2:, 3])) > 5) + 2

            # Adjust shot direction
            dirx = data[i1, 1] - data[0, 1]
            if dirx < 0:
                data[:, [1, 3, 5]] = 2840 - data[:, [1, 3, 5]]

            diry = data[i1, 2] - data[0, 2]
            if diry < 0:
                data[:, [2, 4, 6]] = 1420 - data[:, [2, 4, 6]]

            # Calculate velocity of B1
            B1vel = np.sqrt((data[i1, 1] - data[0, 1])**2 + (data[i1, 2] - data[0, 2])**2) / (data[i1, 0] - data[0, 0])

            # Calculate direction of B1
            B1angle = np.arctan2(data[i1, 2] - data[0, 2], data[i1, 1] - data[0, 1]) * 180 / np.pi

            # Calculate B2 position
            B2pos = [np.mean(data[:i1, 3]), np.mean(data[:i1, 4])]

            # Calculate hit thickness
            dirvec = np.array([data[0, 1], data[0, 2], 0]) - np.array([data[i1 - 3, 1], data[i1 - 3, 2], 0])
            hit_thickness = 1 - np.linalg.norm(np.cross(np.array([B2pos[0], B2pos[1], 0]) - np.array([data[0, 1], data[0, 2], 0]), dirvec)) / np.linalg.norm(dirvec) / 61.5

            # Placeholder for B1 positions and angles
            B1_positions = data[:4, 1:3].flatten()
            B1_angles = [B1angle, np.arctan2(data[3, 2] - data[3, 1], data[3, 1] - data[3, 0]) * 180 / np.pi]

            # Append results
            results.append([file, '', B2pos[0], B2pos[1], hit_thickness, B1vel, 'Y'] +
                           B1_positions.tolist() + B1_angles)

    # Save results to an Excel file
    output_path = os.path.join(directory, "BBT_Evaluation_Results.xlsx")
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False, header=False)
    print(f"Results saved to {output_path}")