from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
from dolfinx.io import gmshio
from dolfinx.fem import locate_dofs_topological
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.optimize import minimize, approx_fprime

# Writing a basic FEM Solver first

def FEM_sol(G,K):

    # Define a function space
    V = fem.functionspace(domain,("Lagrange",1,(3,)))

    # Define the Dirichlet Boundary Conditions
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim,domain.topology.dim)

    """Since the boundary is clamped at one end we will not have any displacements on one end so locating the facet tag 3 and applying 
    boundary conditions"""

    b_D = locate_dofs_topological(V,fdim,facet_tags.find(3))
    u_D = np.array([0,0,0],dtype = default_scalar_type)
    bc_D = fem.dirichletbc(u_D,b_D,V)

    # Define the Neumann Boundary Conditions, i.e. Constant traction on the right hand side of the geometry

    T_neuman = fem.Constant(domain, default_scalar_type((106.26e6,0,0)))
    ds = ufl.Measure("ds",domain=domain,subdomain_data=facet_tags)

    # Define the weak form of our problem
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return  (-2/3 * G + K) * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * G * epsilon(u)

    # Define the test and the trial function
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Apply the Neumann Boundary Conditions at facet 4
    f = fem.Constant(domain,default_scalar_type((0,0,0)))
    a = ufl.inner(sigma(u),epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T_neuman, v) * ds(4)

    # Create Solver Linear
    problem = LinearProblem(a,L,bcs=[bc_D],petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    uh = problem.solve()

    # Post Processing for retreving the solution only in the experimental dataset domain
    x_origin_sub = [20e-3, 100e-3]
    y_origin_sub = 10e-3

    domain_arr = domain.geometry.x

    subdomain_condition = (domain_arr[:, 0] >= x_origin_sub[0]) & (domain_arr[:, 0] <= x_origin_sub[1]) # Extract x-coordinates and apply condition
    num_dofs = V.dofmap.index_map.size_local
    uh_arr = uh.x.array[:].reshape((num_dofs, -1))

    # Load sample points from CSV
    #sample_points_df = pd.read_csv("sample_points.csv")
    #sample_coords = np.array(list(zip(sample_points_df["X"], sample_points_df["Y"])))

    # Interpolate displacement values at sample points
    #u_interp_x = griddata(domain_arr[:, :2], uh_arr[:, 0], sample_coords, method='linear')
    #u_interp_y = griddata(domain_arr[:, :2], uh_arr[:, 1], sample_coords, method='linear')
    #u_interp = np.column_stack((u_interp_x, u_interp_y))


    # Get the subdomain for using with the data
    domain_sub = domain_arr[subdomain_condition]
    domain_sub[:, 0] -= x_origin_sub[0]
    domain_sub[:, 1] += y_origin_sub
    u_sub = uh_arr[subdomain_condition]

    return domain, uh, domain_sub, u_sub
#/root/11257192/
# Interpolating the experimental grid onto the fem grid

def get_sample_points():
    _,_,domain_sub,_ = FEM_sol(73e9,100e9)
    points = domain_sub[:,:2]
    x_min, x_max = 0, 80e-3
    # Select a subset of points for POD (e.g., using uniform sampling)
    num_samples = 100  # Adjust as needed
    indices = np.linspace(0, len(domain_sub) - 1, num_samples, dtype=int)
    sample_points = domain_sub[indices]
    # Save sample points to CSV
    csv_filename = "sample_points_newver.csv"
    df = pd.DataFrame(sample_points, columns=["X", "Y"])
    df.to_csv(csv_filename, index=False)
    # Plot the mesh points and the selected sample points
    plt.figure(figsize=(8, 4))
    plt.scatter(points[:, 0], points[:, 1], s=1, label="Mesh Nodes", alpha=0.3)
    plt.scatter(sample_points[:, 0], sample_points[:, 1], color='red', marker='o', label="Sample Points")
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.title("Sample Points for POD in the Subdomain")
    plt.legend()
    plt.show()


    print(f"Sample points saved to {csv_filename}")
    pass
def Interpolate_exp():
    _,_,node_coordinates,_ = FEM_sol(82e9, 120e9)
    df = pd.read_csv('20231116_displacements_interpolated.csv')
    center_x, center_y = 40e-3, 10e-3
    radius = 4e-3

    def remove_outliers(df,columns,threshold=3):
        cleaned_df = df.copy()
        for column in columns:
            mean = cleaned_df[column].mean()
            std = cleaned_df[column].std()

            z_score = (cleaned_df[column] - mean)/std
            cleaned_df = cleaned_df[np.abs(z_score) <= threshold]
        return cleaned_df
    
    def mask_hole(X,Y):

        # Calculate the distance of each point from the center of the circle
        distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

        # Create a mask: True for points outside the circle, False for points inside
        return distance_from_center >= radius

    def get_displacement_at_point(df,x,y,isClearOutlier):

        if isClearOutlier:
            df = remove_outliers(df,['x','y'],3)  #x, y in mm

            # Extracting points and displacements from the dataframe
            points = df[['x','y']].values/1000
            displacement_x = df['ux'].values/1000
            displacement_y = df['uy'].values/1000

            # Create Interpolators for displacement_x and displacement_y
            interp_x = np.nan_to_num(griddata(points,displacement_x,(x,y),method="linear"),nan=0.0)
            interp_y = np.nan_to_num(griddata(points,displacement_y,(x,y),method="linear"),nan=0.0)

        return interp_x, interp_y

    X,Y = node_coordinates[:,0], node_coordinates[:,1]
    mask = mask_hole(X,Y)
    Z_x, Z_y = get_displacement_at_point(df,X,Y,True)
    X_masked = X[mask]
    Y_masked = Y[mask]
    Z_x_masked = Z_x[mask]
    Z_y_masked = Z_y[mask]

    # Plotting the mesh
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    # Subplot for displacement_x
    contour_x = ax[0].tricontourf(X_masked, Y_masked , Z_x_masked, levels=100, cmap='turbo')
    cbar_x = fig.colorbar(contour_x, ax=ax[0])
    cbar_x.set_label(label='Displacement X (m)', size='18')
    cbar_x.ax.tick_params(labelsize=18)
    ax[0].set_title('Experimental X Displacement',size='20')
    ax[0].set_xlabel('X Coordinate [m]',size='18')
    ax[0].set_ylabel('Y Coordinate [m]', size='18')
    ax[0].tick_params(axis='both', which='major', labelsize=18)
    ax[0].grid(True)

    # Adding the hole to the geometry
    theta = np.linspace(0,2*np.pi,100)
    circle_x = center_x + radius * np.cos(theta)
    circle_y = center_y + radius * np.sin(theta)
    ax[0].fill(circle_x,circle_y,'w-',lw=2)

    # Subplot for Y displacement
    contour_y = ax[1].tricontourf(X_masked, Y_masked , Z_y_masked, levels=100, cmap='turbo')
    cbar_y = fig.colorbar(contour_y, ax=ax[1])
    cbar_y.set_label(label='Displacement Y (m)', size='18')
    cbar_y.ax.tick_params(labelsize=18)
    ax[1].set_title('Experimental Y Displacement', size='20')
    ax[1].set_xlabel('X Coordinate [m]', size='18')
    ax[1].set_ylabel('Y Coordinate [m]', size='18')
    ax[1].tick_params(axis='both', which='major', labelsize=18)
    ax[1].grid(True)

    ax[1].fill(circle_x,circle_y,'w-',lw=2)
#/root/11257192/output_diagram/
    plt.tight_layout()
    plt.savefig('experimental_displacement.pdf',format="pdf")
    return X,Y,Z_x,Z_y

import time
iteration_count = 0
total_start_time = time.time()
def mse_loss(domain,facet_tags,train_ds,beta):
    global iteration_count
    iteration_count += 1  # Increment iteration count
    start_time = time.time() 
    G,K = beta
    _,_,subdomain,subsol = FEM_sol(G,K)
    radius = 4e-3 
    center_x, center_y = 40e-3, 10e-3
    distance_from_center = np.sqrt((subdomain[:, 0] - center_x)**2 + (subdomain[:, 1] - center_y)**2)
    distance_from_center_train = np.sqrt((train_ds[:, 0] - center_x)**2 + (train_ds[:, 1] - center_y)**2)
    train_disp_x = train_ds[:, 2][distance_from_center_train >= radius]
    train_disp_y = train_ds[:, 3][distance_from_center_train >= radius]
    fem_disp_x = subsol[:, 0][distance_from_center >= radius]
    fem_disp_y = subsol[:, 1][distance_from_center >= radius]
    mean_disp_x = np.mean(np.absolute(train_disp_x))
    mean_disp_y = np.mean(np.absolute(train_disp_y))
    Wx = 1/mean_disp_x
    Wy = 1/mean_disp_y
    err_norm_ux = np.linalg.norm(Wx * (train_disp_x - fem_disp_x))
    err_norm_uy = np.linalg.norm(Wy * (train_disp_y - fem_disp_y))
    result = (err_norm_ux**2 + err_norm_uy**2) * 0.5
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Compute elapsed time
    print(beta,result)
    print(f"Iteration {iteration_count}, Beta: {beta}, Loss: {result}, Time Taken: {elapsed_time:.6f} seconds")

    return result

# For plotting the final solution
def mask_hole(X,Y):
    center_x, center_y = 40e-3, 10e-3
    radius = 4e-3
    # Calculate the distance of each point from the center of the circle
    distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

    # Create a mask: True for points outside the circle, False for points inside
    return distance_from_center >= radius

# Plot final displacement solution
def displacement_field(df_solution):
    center_x, center_y = 40e-3, 10e-3
    radius = 4e-3

    def mask_hole(X,Y):
        # Calculate the distance of each point from the center of the circle
        distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

        # Create a mask: True for points outside the circle, False for points inside
        return distance_from_center >= radius
    
    mask = mask_hole(df_solution['x'],df_solution['y'])
    # Plotting the mesh
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    # Subplot for displacement_x
    contour_x = ax[0].tricontourf(df_solution['x'], df_solution['y'], df_solution['z_x'] , levels=100, cmap='turbo')
    cbar_x = fig.colorbar(contour_x, ax=ax[0] )
    cbar_x.set_label(label='Displacement X (m)', size='15')
    cbar_x.ax.tick_params(labelsize=15)
    ax[0].set_title('Computed X Displacement', size='18')
    ax[0].set_xlabel('X Coordinate [m]', size='16')
    ax[0].set_ylabel('Y Coordinate [m]', size='16')
    ax[0].tick_params(axis='both', which='major', labelsize=14)
    ax[0].grid(True)

    # Adding the hole to the geometry
    theta = np.linspace(0,2*np.pi,100)
    circle_x = center_x + radius * np.cos(theta)
    circle_y = center_y + radius * np.sin(theta)
    ax[0].fill(circle_x,circle_y,'w-',lw=2)

    # Subplot for Y displacement
    contour_y = ax[1].tricontourf(df_solution['x'], df_solution['y'], df_solution['z_y'], levels=100, cmap='turbo')
    cbar_y = fig.colorbar(contour_y, ax=ax[1])
    cbar_y.set_label(label='Displacement Y (m)', size='18')
    cbar_y.ax.tick_params(labelsize=18)
    ax[1].set_title('Computed Y Displacement', size='20')
    ax[1].set_xlabel('X Coordinate [m]', size='18')
    ax[1].set_ylabel('Y Coordinate [m]', size='18')
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    ax[1].grid(True)

    ax[1].fill(circle_x,circle_y,'w-',lw=2)
#/root/11257192/output_diagram/
    plt.tight_layout()
    plt.savefig('displacement_field.pdf',format="pdf")
    
    
# Calculating the absolute error between the computed displacements and experimental displacements
def absolute_error(subdomain_sol,deformation_sub_sol,df_solution,X,Y,Z_x,Z_y):
    center_x, center_y = 40e-3, 10e-3
    radius = 4e-3

    def mask_hole(X,Y):
        # Calculate the distance of each point from the center of the circle
        distance_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

        # Create a mask: True for points outside the circle, False for points inside
        return distance_from_center >= radius

    X,Y = subdomain_sol[:,0], subdomain_sol[:,1]
    mask = mask_hole(X,Y)
    error_field_ux = np.abs(Z_x - deformation_sub_sol[:, 0])
    error_field_uy = np.abs(Z_y - deformation_sub_sol[:, 1])
    print(f"The mean error for displacement in x is {np.mean(error_field_ux)}")
    print(f"The mean error for displacement in y is {np.mean(error_field_uy)}")


    # Plotting the mesh
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    # Subplot for displacement_x
    contour_x = ax[0].tricontourf(df_solution['x'], df_solution['y'] , error_field_ux, levels=100, cmap='turbo')
    cbar_x = fig.colorbar(contour_x, ax=ax[0])
    cbar_x.set_label(label='Displacement X (m)', size='18')
    cbar_x.ax.tick_params(labelsize=18)
    ax[0].set_title('Absolute error in X Displacement', size='20')
    ax[0].set_xlabel('X Coordinate [m]', size='18')
    ax[0].set_ylabel('Y Coordinate [m]', size='18')
    ax[0].tick_params(axis='both', which='major', labelsize=18)
    ax[0].grid(True)

    # Adding the hole to the geometry
    theta = np.linspace(0,2*np.pi,100)
    circle_x = center_x + radius * np.cos(theta)
    circle_y = center_y + radius * np.sin(theta)
    ax[0].fill(circle_x,circle_y,'w-',lw=2)

    # Subplot for Y displacement
    contour_y = ax[1].tricontourf(df_solution['x'], df_solution['y'] , error_field_uy, levels=100, cmap='turbo')
    cbar_y = fig.colorbar(contour_y, ax=ax[1])
    cbar_y.set_label(label='Displacement Y (m)', size='18')
    cbar_y.ax.tick_params(labelsize=18)
    ax[1].set_title('Absolute error in Y Displacement',size='20')
    ax[1].set_xlabel('X Coordinate [m]',size='18')
    ax[1].set_ylabel('Y Coordinate [m]',size='18')
    ax[1].tick_params(axis='both', which='major', labelsize=18)
    ax[1].grid(True)

    ax[1].fill(circle_x,circle_y,'w-',lw=2)
#/root/11257192/output_diagram/
    plt.tight_layout()
    plt.savefig('error.pdf',format="pdf")



#/root/11257192/
if __name__ == "__main__":
    domain,cell_tags,facet_tags = gmshio.read_from_msh('tensile_test_specimen.msh', MPI.COMM_WORLD, 0)
    X,Y,Z_x,Z_y = Interpolate_exp()
    train_ds = np.hstack((X.reshape(X.shape[0], 1), Y.reshape(Y.shape[0], 1), Z_x.reshape(Z_x.shape[0], 1), Z_y.reshape(Z_y.shape[0], 1)))
    beta = [50e9, 50e9]
    #counter = 0
    result = minimize(
    fun=lambda x: mse_loss(domain, facet_tags, train_ds, x),
    x0=beta,
    method='Nelder-Mead',
    options={'gtol': 1e-15, 'maxiter': 100}
    )
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    G_sol,K_sol = result.x
    print(f"Final value for Shear Modulus is {G_sol:.2e} & Final value of Bulk Modulus is {K_sol:.2e}")
    print(f"Total number of iterations before convergence was achieved: {result.nit}")
    print(f"\nTotal Optimization Time: {total_elapsed_time:.6f} seconds")
    print(f"Total Iterations: {iteration_count}")
    #print(f"time elapsed: {elapsed_time}")
    #print(f"Iteratiosn done: {iteration_count}")



    domain_sol, deformation_sol, subdomain_sol, deformation_sub_sol = FEM_sol(G_sol,K_sol)
    df_solution = pd.DataFrame({"x": subdomain_sol[:, 0], "y": subdomain_sol[:, 1], "z_x": deformation_sub_sol[:, 0], "z_y": deformation_sub_sol[:, 1]})


    
    displacement_field(df_solution)
    absolute_error(subdomain_sol,deformation_sub_sol,df_solution,X,Y,Z_x,Z_y)
