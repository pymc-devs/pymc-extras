
import pytensor
import pytensor.tensor as pt
from pytensor.graph.replace import vectorize_graph
import numpy as np

def test_vectorize_advanced_inc_subtensor():
    # Create a graph with AdvancedIncSubtensor
    x = pt.matrix("x") # (N, M)
    y = pt.matrix("y") # (K, M)
    idx = [0, 1]
    
    # x[idx] = y
    # This uses AdvancedSetSubtensor (which is AdvancedIncSubtensor with set_instead_of_inc=True)
    z = pt.set_subtensor(x[idx], y)
    
    # Vectorize it
    # We want to batch over a new dimension
    new_x = pt.tensor3("new_x") # (B, N, M)
    new_y = pt.tensor3("new_y")  # (B, K, M)
    
    try:
        new_z = vectorize_graph(z, replace={x: new_x, y: new_y})
        print("Vectorization successful")
        
        # Compile and run to check shapes
        f = pytensor.function([new_x, new_y], new_z)
        
        x_val = np.zeros((1, 5, 5))
        y_val = np.ones((1, 2, 5))
        res = f(x_val, y_val)
        print("Execution successful")
        print("Result shape:", res.shape)
        
    except Exception as e:
        print(f"Vectorization failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vectorize_advanced_inc_subtensor()
