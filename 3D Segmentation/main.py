import itk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import binary_closing, binary_fill_holes
from tkinter import Tk
from tkinter.filedialog import askdirectory
#import matplotlib
#matplotlib.use('TkAgg')

def loadCT(file_path):
    """Load a CT scan using ITK."""
    ct_image = itk.imread(file_path)
    ct_array = itk.array_from_image(ct_image)
    return ct_array, ct_image

# Apply thresholding for segmentation
def segment_lung(ct_array, lower_thresh=-1000, upper_thresh=-300):
    """Segment lung tissue based on intensity thresholds."""
    binary_mask = (ct_array > lower_thresh) & (ct_array < upper_thresh)
    return binary_mask

# Apply morphological operations
def refine_segmentation(binary_mask):
    """Refine the binary mask using morphological operations."""
    # Fill holes
    filled_mask = binary_fill_holes(binary_mask)
    # Close gaps
    refined_mask = binary_closing(filled_mask, structure=np.ones((3, 3, 3)))
    return refined_mask

def visualize_slice_range(original, segmented, start_slice, end_slice):
    """Visualize original and segmented slices for a specified range."""
    for i in range(start_slice, end_slice + 1):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Original Slice {i}")
        plt.imshow(original[i], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title(f"Segmented Slice {i}")
        plt.imshow(segmented[i], cmap="gray")
        plt.show()

# Interactive visualization with a slider
def interactive_visualization(ct_array, segmented_array):
    """Visualize original and segmented slices with a slider."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    slice_index = 0
    original_slice = ct_array[slice_index]
    segmented_slice = segmented_array[slice_index]

    # Initial display
    orig_img = ax[0].imshow(original_slice, cmap="gray")
    ax[0].set_title("Original Slice")
    seg_img = ax[1].imshow(segmented_slice, cmap="gray")
    ax[1].set_title("Segmented Slice")

    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, "Slice", 0, ct_array.shape[0] - 1, valinit=slice_index, valstep=1)

    # Update function for slider
    def update(val):
        slice_index = int(slider.val)  # Get the current slider value
        orig_img.set_data(ct_array[slice_index])  # Update original slice image
        seg_img.set_data(segmented_array[slice_index])  # Update segmented slice image
        fig.canvas.draw_idle()  # Redraw the canvas

    slider.on_changed(update)
    plt.show()


# Save segmented mask
def save_segmented_mask(segmented_array, reference_image, output_path):
    """Save the segmented mask as an ITK image."""
    segmented_image = itk.image_from_array(segmented_array.astype(np.uint8))
    segmented_image.SetSpacing(reference_image.GetSpacing())
    segmented_image.SetOrigin(reference_image.GetOrigin())
    segmented_image.SetDirection(reference_image.GetDirection())
    itk.imwrite(segmented_image, output_path)

# Main workflow
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

    binary_mask = segment_lung(ctArray)

    refined_mask = refine_segmentation(binary_mask)

    print("Enter the range of slices to visualize:")
    start_slice = int(input("Start slice (0-indexed): "))
    end_slice = int(input("End slice (0-indexed): "))
    if start_slice < 0 or end_slice >= ctArray.shape[0] or start_slice > end_slice:
        print("Invalid range. Please enter valid slice indices.")
        return

    visualize_slice_range(ctArray, refined_mask, start_slice, end_slice)

    #interactive_visualization(ctArray, refined_mask)

    # Step 4: Save Results
    output_path = "segmented_lung.nii.gz"
    save_segmented_mask(refined_mask, ctImage, output_path)
    print(f"Segmentation saved as '{output_path}'.")

if __name__ == "__main__":
    main()
