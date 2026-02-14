import dv_processing as dv
import matplotlib.pyplot as plt
import numpy as np 

def Get_Covariance(X_Points, Y_Points):
    X_Mean = np.mean(X_Points)
    Y_Mean = np.mean(Y_Points)
    return np.sum((X_Points - X_Mean) * (Y_Points - Y_Mean)) / (np.size(X_Points) - 1)

def Get_PCA(X_Points, Y_Points):
    # Calculate Covariances
    Cov_XX = Get_Covariance(X_Points, X_Points)
    Cov_YY = Get_Covariance(Y_Points, Y_Points)
    Cov_XY = Get_Covariance(X_Points, Y_Points)

    # Calculate Eigenvalues by using the Quadratic Equation
    a = 1.0
    b = -(Cov_XX + Cov_YY)
    c = (Cov_XX * Cov_YY) - (Cov_XY**2)
    Lambda_1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    Lambda_2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

    # 4. Calculate Eigenvectors 
    # v = [lambda - Cov(Y,Y), Cov(X,Y)]
    Vector_1 = np.array([Lambda_1 - Cov_YY, Cov_XY])
    Vector_2 = np.array([Lambda_2 - Cov_YY, Cov_XY])

    return (Lambda_1, Lambda_2), (Vector_1, Vector_2)

Camera = dv.io.MonoCameraRecording("Ruler.aedat4")
Times_Middle = []
Times = []
X_Data = []
X_Centers = []
Y_Data = []
Y_Centers = []
Vector_U = []
Vector_V = []
while Camera.isRunning():
    # 1. Read Events
    if Camera.isEventStreamAvailable():
        Events = Camera.getNextEventBatch()
        if Events is not None and Events.size() > 0:
            # Structure fields: ["timestamp", "x", "y", "polarity"]
            Data = Events.numpy() 
            Times.extend(Data["timestamp"])
            X_Data.extend(Data["x"])
            Y_Data.extend(Data["y"])

# Convert Lists to Numpy arrays
Times = np.array(Times)
Times = Times - Times[0]
X_Data = np.array(X_Data)
Y_Data = np.array(Y_Data)

# Define window and step sizes
Window_Size = 5000
Step_Size = Window_Size//2

# Loop through the data and filter
Times_Clean = []
X_Clean = []
Y_Clean = []
for i in range(0, len(Times) - Window_Size + 1, Step_Size):
    # Slice the array to get the current window_size points
    Window_X = X_Data[i : i + Window_Size]
    Window_Y = Y_Data[i : i + Window_Size]
    Window_Times = Times[i : i + Window_Size]

    # Calculate the means of the data and push to center lists
    Mean_X, Mean_Y = np.mean(Window_X), np.mean(Window_Y)
    X_Centers.append(Mean_X)
    Y_Centers.append(Mean_Y)
    Times_Middle.append(np.median(Window_Times))

    # Get PCA for the window and get the unit eigen vectors
    (Lambda_1, Lambda_2), (V1, V2) = Get_PCA(Window_X, Window_Y)
    U_Unit = V1 / np.linalg.norm(V1)
    V_Unit = V2 / np.linalg.norm(V2)
    Vector_U.append(U_Unit * np.sqrt(Lambda_1))
    Vector_V.append(V_Unit * np.sqrt(Lambda_2))

    # Remove the points that are not beyond 
    Displacement_X = Window_X- Mean_X
    Displacement_Y = Window_Y - Mean_Y
    Num_Sigma_Sq = ((Displacement_X * U_Unit[0] + Displacement_Y * U_Unit[1])**2 / Lambda_1) + \
               ((Displacement_X * V_Unit[0] + Displacement_Y * V_Unit[1])**2 / Lambda_2)
    Indexes_Remove = np.sqrt(Num_Sigma_Sq) < 1 # Remove points that are beyond the first standard deviation
    Times_Clean.extend(Window_Times[Indexes_Remove])
    X_Clean.extend(Window_X[Indexes_Remove])
    Y_Clean.extend(Window_Y[Indexes_Remove])

# Create 3-D Scatter plot of the filtered data
Figure = plt.figure(figsize = (10, 8))
Axe = Figure.add_subplot(111, projection = "3d")
Axe.set_title(f"PCA Event Stream 3D Visualization")
Axe.set_xlabel("Time")
Axe.set_ylabel("X")
Axe.set_zlabel("Y")
Axe.scatter(Times_Middle, X_Centers, Y_Centers, s = 5, alpha = 0.5)

# PCA vectors 
Times_Middle = Times_Middle - Times_Middle[0]
Vector_U = np.array(Vector_U)
Vector_V = np.array(Vector_V)
'''
Axe.quiver(Times_Middle, X_Centers, Y_Centers, 
          np.zeros_like(Times_Middle), Vector_U[:,0], Vector_U[:,1],
          length = 1, normalize = False, color = "red")
Axe.quiver(Times_Middle, X_Centers, Y_Centers, 
          np.zeros_like(Times_Middle), Vector_V[:,0], Vector_V[:,1],
          length = 1, normalize = False, color = "blue")
'''

plt.savefig("Filtered_PCA_Data.png")
plt.show()