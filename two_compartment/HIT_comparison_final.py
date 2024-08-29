import numpy as np
import matplotlib.pyplot as plt

beta_1 = 2
beta_2 = 2
gamma_1 = 1
gamma_2 = 1
epsilon_list = [0.001,0.1,0.4]

def HIT(beta_1,beta_2,gamma_1,gamma_2,epsilon,h_1,h_2):
    a = (beta_1 * (1-h_1)) / (gamma_1 )
    b = (beta_2 * (1-h_2)) / (gamma_2 )
    r = (1 - epsilon) * (a + b) / 2 + np.sqrt(
            (1 - epsilon) ** 2 * (a + b) ** 2 - 4 * (1 - 2 * epsilon) * a * b
        ) / 2
    return r

def get_most_unstable_subspace(beta_1,beta_2,gamma_1,gamma_2,epsilon,h_1,h_2):
    a = (beta_1 * (1-h_1)) / (gamma_1 )
    b = (beta_2 * (1-h_2)) / (gamma_2 )
    next_generation_matrix = np.array([[(1-epsilon) * a, epsilon * b],[epsilon * a, (1-epsilon)* b ]])
    # obtain the eigenvector corresponding to the largest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eig(next_generation_matrix)
    # obtain the index of the largest eigenvalue
    max_eigenvalue_index = np.argmax(eigenvalues)
    # obtain the eigenvector corresponding to the largest eigenvalue
    max_eigenvector = eigenvectors[:,max_eigenvalue_index]
    return max_eigenvector

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

for k,epsilon in enumerate(epsilon_list):
    # make a heatmap of HIT() against h_1 and h_2
    h_1 = np.linspace(0,1,100)
    h_2 = np.linspace(0,1,100)
    HIT_values = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            HIT_values[i,j] = HIT(beta_1,beta_2,gamma_1,gamma_2,epsilon,h_1[i],h_2[j])
    # plot the heatmap with colormap bwr
    im = ax[k].imshow(HIT_values,origin='lower',extent=[0,1,0,1],cmap='bwr')
    # add label to the colorbar
    # draw the line representing h_1 + h_2 = 0.5
    ax[k].plot(h_1,1-h_1,'k',linestyle="--",label=r'$h_{total} = 0.5$')
    ax[k].set_xlim(0,1)
    ax[k].set_ylim(0,1)
    # drawing the contour line for HIT = 1
    contour = ax[k].contour(h_1,h_2,HIT_values,levels=[1,1.5],colors=['b','b'],linestyles=["-","--"])
    if k != 2:
        manual_positions = [(0.8, 0.5),(0.6,0.3)]  # Example positions
    else:
        manual_positions = [(0.8, 0.5),(0.4,0.2)]
    ax[k].clabel(contour, fmt={1: r'$R_{eff} = 1$',1.5: r'$R_{eff} = 1.5$'}, manual=manual_positions, fontsize=15)
    ax[k].set_xlabel(r'$h_1$',fontsize=15)
    ax[k].set_title(r'$\epsilon = {}$'.format(epsilon),fontsize=15)
    if k == 0:
        ax[k].set_ylabel(r'$h_2$',fontsize=15)
    ax[k].legend(fontsize=15)
    h_1_sample_points = np.linspace(0.1,0.9,5)
    h_2_sample_points = np.linspace(0.1,0.9,5)
    """for h_1_sample_point in h_1_sample_points:
        for h_2_sample_point in h_2_sample_points:
            max_eigenvector = get_most_unstable_subspace(beta_1,beta_2,gamma_1,gamma_2,epsilon,h_1_sample_point,h_2_sample_point)
            HIT_value = HIT(beta_1,beta_2,gamma_1,gamma_2,epsilon,h_1_sample_point,h_2_sample_point)
            if HIT_value > 1:
                # annotate the eigenvalue
                ax[k].annotate(f"{HIT_value:.2f}",(h_1_sample_point,h_2_sample_point))
                print(f"(h_1,h_2)=({h_1_sample_point:.2f},{h_2_sample_point:.2f})","max eigenvector",max_eigenvector)
                ax[k].quiver(h_1_sample_point,h_2_sample_point,max_eigenvector[0],max_eigenvector[1],scale=10,color='k')"""

# Get the position of the first row of subplots
pos0 = ax[0].get_position()
pos1 = ax[1].get_position()
pos2 = ax[2].get_position()

# Calculate the coordinates for the color bar axis
cbar_x = pos2.x1 + 0.01
cbar_y = pos2.y0
cbar_width = 0.02
cbar_height = pos0.y1 - pos2.y0

# Add a single color bar for the first row subplots
cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(r'$R_{eff}$',fontsize=15)
plt.savefig('/Users/bouningen0909/dissertation/week_4/data/HIT_heatmap.png',dpi=300)
plt.show()