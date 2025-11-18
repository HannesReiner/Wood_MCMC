import pandas as pd
import numpy as np

# Read the data from the two CSV files
file1_path = "InputData219_WoodLSTM4_90.csv"
file2_path = "InputData219_WoodLSTM45.csv"

data1 = pd.read_csv(file1_path, header=None)
data2 = pd.read_csv(file2_path, header=None)

# Combine the two datasets
#combined_data = pd.concat([data1, data2], ignore_index=True)
combined_data = pd.concat([data1, data2], axis=1, ignore_index=True)

#print(combined_data)

# Rename columns for clarity
combined_data.columns = ["E1", "eps1i", "eps1s", "E2", "G12", "eps2i", "eps2s"]

# Add constant Poisson's ratio and thickness
combined_data["v12"] = 0.073  # Constant Poisson's ratio
combined_data["thickness"] = 0.5  # Constant thickness
#combined_data["interlaminar_strength"] = 200  # Constant thickness

# Select only the required columns
selected_data = combined_data[["E1", "E2", "G12", "v12", "thickness"]]
#print(selected_data)

# Define the stacking sequence (angles in degrees)
stacking_sequence = [
    {"angle": 90},
    {"angle": 45},
    {"angle": 0},
    {"angle": -45},
    {"angle": -45},
    {"angle": 0},
    {"angle": 45},
    {"angle": 90},
]

# Initialize a list to store the laminate modulus for each row
laminate_moduli = []

# Loop over each row of the input data
for index, row in selected_data.iterrows():
    # Extract material properties for the current row
    E1 = row["E1"]
    #print(E1)
    E2 = row["E2"]
    #print(E2)
    G12 = row["G12"]
    #print(G12)
    v12 = row["v12"]
    #print(v12)
    thickness = row["thickness"]
    #print(thickness)
    
    # Calculate total thickness of the laminate
    total_thickness = thickness * len(stacking_sequence)


    # Calculate total thickness and ply boundaries
    z = [0]  # Start at 0
    for ply in stacking_sequence:
        z.append(z[-1] + thickness)  # Add thickness of each ply

    # Initialize A matrix (only A is needed for laminate modulus calculation)
    A = np.zeros((3, 3))

    # Loop through each ply to calculate the A matrix
    for i in range(len(stacking_sequence)):
        ply = stacking_sequence[i]
        angle = ply["angle"]

        # Calculate Q matrix for the ply
        Q11 = E1 / (1 - v12 * (E2 / E1))
        Q22 = E2 / (1 - v12 * (E2 / E1))
        Q12 = v12 * E2 / (1 - v12 * (E2 / E1))
        Q66 = G12

        Q = np.array([[Q11, Q12, 0],
                      [Q12, Q22, 0],
                      [0,    0,  Q66]])

        # Transform Q matrix based on ply angle
        theta = np.radians(angle)  # Convert angle to radians
        m = np.cos(theta)
        n = np.sin(theta)
        T = np.array([[m**2, n**2, 2*m*n],
                      [n**2, m**2, -2*m*n],
                      [-m*n, m*n, m**2 - n**2]])

        # Transformed Q matrix
        Q_bar = np.dot(np.dot(T.T, Q), T)

        # Update A matrix
        A += Q_bar * (z[i + 1] - z[i])
    # Calculate the inverse of the A matrix
    A_inv = np.linalg.inv(A)
    # Calculate the laminate modulus (Ex) from the A matrix
    #print(A[0,0])
    laminate_modulus = 1 / (A_inv[0, 0]*total_thickness)  # Ex = 1 / A11
    laminate_moduli.append(laminate_modulus)

    # Print progress for debugging
    print(f"Row {index + 1}: Laminate Modulus (Ex) = {laminate_modulus:.6f}")

# Store the results in a DataFrame and save to a CSV file
results_df = pd.DataFrame({"Laminate Modulus (Ex)": laminate_moduli})
results_df.to_csv("Laminate_Moduli.csv", index=False)

print(np.mean(laminate_moduli))
print(np.std(laminate_moduli))
# Print the results
#print("Laminate moduli for all rows:")
#print(results_df)
