
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


class Node:
    def __init__(self, x, y, z, is_fixed=False):
        self.pos = np.array([x, y, z], dtype=float)
        self.vel = np.zeros(3)
        self.force = np.zeros(3)
        self.is_fixed = is_fixed
        
class Spring:
    def __init__(self, node1, node2, k, rest_length=None):
        self.node1 = node1
        self.node2 = node2
        self.k = k
        if rest_length is None:
            self.rest_length = np.linalg.norm(node1.pos - node2.pos)
        else:
            self.rest_length = rest_length

def create_3x3_grid(size=1.0, k=2):
    # Create nodes
    nodes = []
    for i in range(3):
        for j in range(3):
            x = j * size
            y = i * size
            z = 0.0
            # Corner nodes are fixed
            is_fixed = (i in [0, 2] and j in [0, 2])
            nodes.append(Node(x, y, z, is_fixed))
    
    # Create springs (only orthogonal connections)
    springs = []
    # Horizontal and vertical springs
    for i in range(9):
        # Horizontal connections
        if i % 3 < 2:
            springs.append(Spring(nodes[i], nodes[i + 1], k))
        # Vertical connections
        if i < 6:
            springs.append(Spring(nodes[i], nodes[i + 3], k))
    
    return nodes, springs

def dynamic_relaxation(nodes, springs, center_load=2, max_iterations=3000,
                      damping=0.98, dt=0.01, tolerance=1e-6):
    iterations = []
    energies = []
    center_node_z = []
    residual_forces = []  
    
    # Apply load to center node
    nodes[4].force[2] = center_load
    
    for iteration in range(max_iterations):
        # Reset forces on nodes (except fixed load)
        for node in nodes:
            if not node.is_fixed:
                node.force = np.zeros(3)
                node.force[2] = center_load if node == nodes[4] else 0
        
        # Calculate spring forces
        total_energy = 0
        for spring in springs:
            direction = spring.node2.pos - spring.node1.pos
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction_unit = direction / distance
                force_magnitude = spring.k * (distance - spring.rest_length)
                force = force_magnitude * direction_unit
                
                if not spring.node1.is_fixed:
                    spring.node1.force += force
                if not spring.node2.is_fixed:
                    spring.node2.force -= force
                    
                # Calculate spring potential energy
                total_energy += 0.5 * spring.k * (distance - spring.rest_length)**2
        
        # Update velocities and positions
        max_force = 0
        for node in nodes:
            if not node.is_fixed:
                # Update velocity with damping
                node.vel += node.force * dt
                node.vel *= damping
                
                # Update position
                node.pos += node.vel * dt
                
                # Track maximum force for convergence check
                max_force = max(max_force, np.linalg.norm(node.force))
        
        # Store iteration data
        iterations.append(iteration)
        energies.append(total_energy)
        center_node_z.append(nodes[4].pos[2])
        residual_forces.append(max_force)  # Store the maximum residual force
        
        # Check convergence
        if max_force < tolerance:
            break
    
    return iterations, energies, center_node_z, residual_forces

def plot_grid(nodes, springs, ax=None, show=True):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot springs
    for spring in springs:
        x = [spring.node1.pos[0], spring.node2.pos[0]]
        y = [spring.node1.pos[1], spring.node2.pos[1]]
        z = [spring.node1.pos[2], spring.node2.pos[2]]
        ax.plot(x, y, z, 'b-', linewidth=2)
    
    # Plot nodes
    for node in nodes:
        color = 'red' if node.is_fixed else 'blue'
        ax.scatter(node.pos[0], node.pos[1], node.pos[2], 
                  color=color, s=100)
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3x3 Orthogonal Grid Form-finding')
    
    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1,1,1])
    
    if show:
        plt.show()

# Main execution
if __name__ == "__main__":
    # Create grid
    nodes, springs = create_3x3_grid(size=1.0, k=2)
    
    # Run simulation
    iterations, energies, center_z, residual_forces = dynamic_relaxation(
        nodes, springs, 
        center_load=2,
        max_iterations=3000,
        damping=0.98,
        dt=0.01,
        tolerance=1e-5
    )
    
    # Plot final configuration with z-coordinate annotations
    def plot_grid_with_annotations(nodes, springs, ax=None, show=True):
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot springs
        for spring in springs:
            x = [spring.node1.pos[0], spring.node2.pos[0]]
            y = [spring.node1.pos[1], spring.node2.pos[1]]
            z = [spring.node1.pos[2], spring.node2.pos[2]]
            ax.plot(x, y, z, 'b-', linewidth=2)
        
        # Plot nodes
        for node in nodes:
            color = 'red' if node.is_fixed else 'blue'
            ax.scatter(node.pos[0], node.pos[1], node.pos[2], color=color, s=100)
            # Annotate nodes with their z-coordinate
            ax.text(node.pos[0], node.pos[1], node.pos[2], 
                    f'{node.pos[2]:.2f}', color='black', fontsize=10)
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3x3 Orthogonal Grid Form-finding')
        
        # Set equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])
        
        if show:
            plt.show()
    
    plot_grid_with_annotations(nodes, springs)
    
    # Plot convergence history with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Energy convergence plot
    ax1.plot(iterations, energies)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Energy')
    ax1.set_title('Energy Convergence')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # Center node displacement plot
    ax2.plot(iterations, center_z)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Center Node Z-Position [cm]')
    ax2.set_title('Center Node Displacement [cm]')
    ax2.grid(True)
    
    # Residual force plot
    ax3.plot(iterations, residual_forces)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Maximum Residual Force [N]')
    ax3.set_title('Residual Force Convergence')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\nForm-finding Results:")
    print("================================")
    print(f"Final center node position: {nodes[4].pos}")
    print(f"Number of iterations: {len(iterations)}")
    print(f"Final energy: {energies[-1]:.6e}")
    print(f"Final residual force: {residual_forces[-1]:.6e}")
    print("================================")