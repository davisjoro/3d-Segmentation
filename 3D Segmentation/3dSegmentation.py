import itk
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes
import pyvista as pv
import matplotlib
from tkinter import Tk
from tkinter.filedialog import askdirectory

#matplotlib.use('TkAgg')

def loadCT(folderPath):
    ctImage = itk.imread(folderPath)
    ctArray = itk.array_from_image(ctImage)
    return ctArray, ctImage

def segLung(ctArray, lowThresh, upperThresh):
    binaryMask = (ctArray > lowThresh) & (ctArray < upperThresh)
    return binaryMask

def refineSeg(binary_mask):
    filledMask = binary_fill_holes(binary_mask)
    refinedMask = binary_closing(filledMask, structure=np.ones((3, 3, 3)))
    return refinedMask

def combineTo3d(slices):
    volume = np.stack(slices, axis=0)
    return volume

def visualize3d(volume, numSlices):
    scan = pv.UniformGrid()
    scan.dimensions = np.array(volume.shape) + 1
    scan.origin = (0, 0, 0)
    scan.spacing = (230/numSlices, 0.5, 0.5)
    scan.cell_data["values"] = volume.flatten(order="F")

    # render
    plotter = pv.Plotter()
    plotter.add_volume(scan, cmap="gray", opacity="linear")
    plotter.show()

def main():
    Tk().withdraw()  # Hide the root window of tkinter
    folderPath = askdirectory(
        title="Select a Folder Containing CT Scans"
    )

    if not folderPath:
        print("No folder selected. Exiting...")
        return
    # Replace with your file path
    ctArray, ctImage = loadCT(folderPath)

    print("CT scan shape:", ctArray.shape)
    print("Enter intensity thresholds for lung segmentation:")
    lowThresh = float(input("Lower threshold (default -1000): ") or -1000)
    upperThresh = float(input("Upper threshold (default -300): ") or -300)

    # Thresholding
    binary_mask = segLung(ctArray, lowThresh, upperThresh)

    # Morphological Refinement
    refined_mask = refineSeg(binary_mask)

    # Combine into 3D
    combined = combineTo3d(refined_mask)
    combined2 = combineTo3d(binary_mask)

    #visualize3d(combined2, ctArray.shape[0])
    #visualize3d(ctArray, ctArray.shape[0])
    visualize3d(combined, ctArray.shape[0])



if __name__ == "__main__":
    main()
