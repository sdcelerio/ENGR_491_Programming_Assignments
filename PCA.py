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
            Eigen_Vals, Eigen_Vec = Get_PCA(Data["x"], Data["y"])

            mu = np.array([np.mean(Data["x"]), np.mean(Data["y"])])
            cov = np.cov(Data["x"], Data["y"])
            vals, vecs = np.linalg.eigh(cov)
            Times.extend(Data["timestamp"])
            X_Data.extend(Data["x"])
            Y_Data.extend(Data["y"])
            Times_Middle.append(np.median(Data["timestamp"]))
            X_Centers.append(np.mean(Data["x"]))
            Y_Centers.append(np.mean(Data["y"]))
            Vector_U.append(Eigen_Vec[0]/np.linalg.norm(Eigen_Vec[0])) 
            Vector_V.append(Eigen_Vec[1]/np.linalg.norm(Eigen_Vec[1]))

# Convert Lists to Numpy arrays
Times = np.array(Times)
Times = Times - Times[0]
X_Data = np.array(X_Data)
X_Centers = np.array(X_Centers)
Y_Data = np.array(Y_Data)
Y_Centers = np.array(Y_Centers)
Vector_U = np.array(Vector_U)
Vector_V = np.array(Vector_V)

# Create 3-D Scatter plot of the filtered data
Figure = plt.figure(figsize = (10, 8))
Axe = Figure.add_subplot(111, projection = "3d")
Axe.set_title(f"PCA Event Stream 3D Visualization")
Axe.set_xlabel("Time")
Axe.set_ylabel("X")
Axe.set_zlabel("Y")

# Scatter Plot
Times_Middle = Times_Middle - Times_Middle[0]
Axe.scatter(Times[::1000], X_Data[::1000], Y_Data[::1000], s = 5)

# PCA vectors 
Axe.quiver(Times_Middle, X_Centers, Y_Centers, 
          np.zeros_like(Times_Middle), Vector_U[:,0], Vector_U[:,1],
          length = 50, normalize = False, color = "red")
Axe.quiver(Times_Middle, X_Centers, Y_Centers, 
          np.zeros_like(Times_Middle), Vector_V[:,0], Vector_V[:,1],
          length = 50, normalize = False, color = "blue")

plt.show()