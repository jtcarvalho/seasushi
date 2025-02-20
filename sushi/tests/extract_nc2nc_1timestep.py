import os
from sushi.tools import save_time_step
# import sys
# print(sys.path)

def main():
    input_file = "../../../soap/soap/data/ww3-glo025.nc"
    output_file = "./data/ww3_global_example.nc"
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Extract and save the last time step
    save_time_step(input_file, output_file)

if __name__ == "__main__":
    main()
