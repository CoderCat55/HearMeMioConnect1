"""
This script aims to plot specified folders along with some grahical requirenments

Constants:
main_folder_path= ""
subfolders_list = ["p1","p2","p3","p4","p5","p6"]
plots_folder =""
Xgrid_size= 5
Ygrid_size= 100
legant =[ItuEMG1, ItuEMG2, ItuEMG3, ItuEMG4, ItuEMG5, ItuEMG6, ItuEMG7, ItuEMG8, ItuROLL, ItuPITCH, ItuYAW, ItuAX, ItuAY, ItuAZ,
 ItuGX, ItuGY, ItuGZ, MarEMG1, MarEMG2, MarEMG3, MarEMG4, MarEMG5, MarEMG6, MarEMG7, MarEMG8, MarROLL, MarPITCH,
   MarYAW, MarAX, MarAY, MarAZ, MarGX, MarGY, MarGZ] #from column 0 to 33 in npy files

Graphical requirenments:
x values should go from  0 to 300 with 50 unit interval
y values should go from 6000 to -6000 with 1000 unit interval
graphic name should be {subfoldername}+{nameofnpyfile}
has grids specifies lengts : Xgrid_size,Ygrid_size grid lines should be visible enough to see without magnifiyng the image
include a legant
background should be black

Flow:
for a in range subfolders_list.length
    path=main_folder_path+subfolders_list[a]
    append all .npy files to a list 
    use matplotlib to plot the npy files
    create a folder inside plots_folder named subfolders_list[a]
    save this plot into path=plots_folder+subfolders_list[a]
    report any erors
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
main_folder_path = "lastcb"  # Set your main folder path here
subfolders_list = [""]
plots_folder = "lastcb_plots"  # Set your plots output folder here
Xgrid_size = 5
Ygrid_size = 100

# Legend labels (columns 0-33)
legant = [
    "ItuEMG1", "ItuEMG2", "ItuEMG3", "ItuEMG4", "ItuEMG5", "ItuEMG6", "ItuEMG7", "ItuEMG8",
    "ItuROLL", "ItuPITCH", "ItuYAW", "ItuAX", "ItuAY", "ItuAZ", "ItuGX", "ItuGY", "ItuGZ",
    "MarEMG1", "MarEMG2", "MarEMG3", "MarEMG4", "MarEMG5", "MarEMG6", "MarEMG7", "MarEMG8",
    "MarROLL", "MarPITCH", "MarYAW", "MarAX", "MarAY", "MarAZ", "MarGX", "MarGY", "MarGZ"
]


def plot_npy_file(npy_path, save_path, subfolder_name):
    """
    Plot a single .npy file with specified graphical requirements.
    
    Args:
        npy_path: Path to the .npy file
        save_path: Path where the plot should be saved
        subfolder_name: Name of the subfolder for the plot title
    """
    try:
        # Load the .npy file
        data = np.load(npy_path)
        
        # Get the filename without extension
        npy_filename = Path(npy_path).stem
        
        # Create figure with black background
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='black')
        ax.set_facecolor('black')
        
        # Plot each column
        num_columns = min(data.shape[1] if len(data.shape) > 1 else 1, len(legant))
        
        for col in range(num_columns):
            if len(data.shape) > 1:
                ax.plot(data[:, col], label=legant[col], linewidth=0.8)
            else:
                ax.plot(data, label=legant[0], linewidth=0.8)
                break
        
        # Set x-axis: 0 to 300 with 50 unit intervals
        ax.set_xlim(0, 300)
        ax.set_xticks(np.arange(0, 301, 50))
        
        # Set y-axis: -6000 to 6000 with 1000 unit intervals
        ax.set_ylim(-6000, 6000)
        ax.set_yticks(np.arange(-6000, 6001, 1000))
        
        # Add grid with specified sizes
        ax.grid(True, color='white', alpha=0.5, linewidth=0.5)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(Xgrid_size))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(Ygrid_size))
        ax.grid(which='minor', color='white', alpha=0.40, linewidth=0.3)
        
        # Set labels and title with white color
        ax.set_xlabel('Time', color='white', fontsize=10)
        ax.set_ylabel('Value', color='white', fontsize=10)
        ax.set_title(f'{subfolder_name}_{npy_filename}', color='white', fontsize=12, pad=10)
        
        # Customize tick colors
        ax.tick_params(colors='white', which='both')
        
        # Add legend with white text on dark background
        legend = ax.legend(
            loc='upper right',
            fontsize=6,
            ncol=2,
            facecolor='#1a1a1a',
            edgecolor='white',
            framealpha=0.8
        )
        for text in legend.get_texts():
            text.set_color('white')
        
        # Save the plot
        save_filename = f'{subfolder_name}_{npy_filename}.png'
        full_save_path = os.path.join(save_path, save_filename)
        plt.tight_layout()
        plt.savefig(full_save_path, facecolor='black', dpi=150)
        plt.close(fig)
        
        print(f"✓ Successfully saved: {save_filename}")
        
    except Exception as e:
        print(f"✗ Error processing {npy_path}: {str(e)}")


def main():
    """Main function to process all subfolders and plot .npy files."""
    
    # Validate paths
    if not main_folder_path:
        print("Error: Please set 'main_folder_path' variable")
        return
    
    if not plots_folder:
        print("Error: Please set 'plots_folder' variable")
        return
    
    # Create main plots folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)
    
    # Process each subfolder
    for subfolder in subfolders_list:
        print(f"\n{'='*60}")
        print(f"Processing subfolder: {subfolder}")
        print(f"{'='*60}")
        
        # Construct path to subfolder
        subfolder_path = os.path.join(main_folder_path, subfolder)
        
        # Check if subfolder exists
        if not os.path.exists(subfolder_path):
            print(f"✗ Warning: Subfolder not found: {subfolder_path}")
            continue
        
        # Create output folder for this subfolder
        output_folder = os.path.join(plots_folder, subfolder)
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder: {output_folder}")
        
        # Find all .npy files in the subfolder
        npy_files = []
        try:
            for file in os.listdir(subfolder_path):
                if file.endswith('.npy'):
                    npy_files.append(os.path.join(subfolder_path, file))
        except Exception as e:
            print(f"✗ Error reading subfolder {subfolder}: {str(e)}")
            continue
        
        if not npy_files:
            print(f"✗ No .npy files found in {subfolder}")
            continue
        
        print(f"Found {len(npy_files)} .npy file(s)")
        
        # Plot each .npy file
        for npy_file in npy_files:
            plot_npy_file(npy_file, output_folder, subfolder)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()