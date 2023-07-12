# from pydicom import dcmread
# import png
# import numpy as np
# from pydicom.pixel_data_handlers.util import apply_voi_lut

# #file = './data/PseudoID0001/DX/Tomosynthesis/PseudoID0001 Image 001 Slice location 20,00.dcm'
# file = './data/PseudoID0001/CT/Thorax insp/PseudoID0001 Image 001 Slice location -752,50.dcm'

# ds = dcmread(file)
# if 'WindowWidth' in ds:
#     print('Dataset has windowing')

# # windowed = apply_voi_lut(ds.pixel_array, ds)

# shape = ds.pixel_array.shape

# # Convert to float to avoid overflow or underflow losses.
# image_2d = ds.pixel_array.astype(float)

# # Rescaling grey scale between 0-255
# image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

# # Convert to uint
# image_2d_scaled = np.uint8(image_2d_scaled)

# # Write the PNG file
# with open('output.png', 'wb') as png_file:
#     w = png.Writer(shape[1], shape[0], greyscale=True)
#     w.write(png_file, image_2d_scaled)


# # Add code for rescaling to 8-bit...

import os
import pydicom
import numpy as np
import png
from PIL import Image, ImageOps


def resize_and_crop_center(img, new_width=512, new_height=512):
    w, h = img.size

    # Only downscale if needed.
    if w > new_width or h > new_height:
        # Use thumbnail() function to resize the input image while maintaining the aspect ratio
        img.thumbnail((new_width, new_height), Image.LANCZOS)

    # The crop function takes in a tuple of the left, upper, right, and lower pixel
    # coordinates, and returns a rectangular region from the used image.
    left = (img.width - new_width)/2
    top = (img.height - new_height)/2
    right = (img.width + new_width)/2
    bottom = (img.height + new_height)/2

    # Ensure dimensions are integer
    left, top, right, bottom = round(left), round(
        top), round(right), round(bottom)

    img = img.crop((left, top, right, bottom))

    return img


def transform_tomosynthesis(inputdir, outdir):
    # Create output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Create an empty list and dict

    # loop through all the DICOM files
    i = 0
    for filename in os.listdir(inputdir):
        if filename.endswith(".dcm"):
            ds = pydicom.read_file(os.path.join(inputdir, filename))
            # # windowed = apply_voi_lut(ds.pixel_array, ds)

            # Convert to float to avoid overflow or underflow losses.
            image_2d = ds.pixel_array.astype(float)

            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d, 0) /
                               image_2d.max()) * 255.0

            # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)

            # Create PIL image
            img = Image.fromarray(image_2d_scaled)

            # Resize and crop
            img = resize_and_crop_center(img, 512, 512)

            # Save image as 3 channel RGB
            img = ImageOps.grayscale(img)
            img = img.convert('RGB')

            img.save(os.path.join(outdir, f"im{i}.png"))
            i += 1


def transform_ct(inputdir, outdir):
    # Create output directory if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Create an empty list and dict
    lstFilesDCM = []
    slices = []

    # Get ref file
    RefDs = pydicom.dcmread(os.path.join(inputdir, os.listdir(inputdir)[0]))

    # Load dimensions based on the number of rows, columns, and slices
    ConstPixelDims = (int(RefDs.Rows), int(
        RefDs.Columns), len(os.listdir(inputdir)))

    # Load spacing values
    # ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(
    #    RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # Create a 3D array to hold the pixel data
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filename in os.listdir(inputdir):
        if filename.endswith(".dcm"):
            # read the file
            ds = pydicom.read_file(os.path.join(inputdir, filename))
            # store the raw image data and the slice location
            lstFilesDCM.append(filename)
            ArrayDicom[:, :, lstFilesDCM.index(filename)] = ds.pixel_array
            slices.append((ds.SliceLocation, ds.pixel_array))

    # Sort the slices by location
    sorted_slices = np.array(
        [x[1] for x in sorted(slices, key=lambda x: x[0])])

    # Convert to uint8
    rescaled_image = ((sorted_slices - np.min(sorted_slices)) /
                      (np.max(sorted_slices) - np.min(sorted_slices)) * 255.0).astype(np.uint8)

    # Transpose the array to get coronal view, and flip up-down
    rescaled_image = np.flip(np.transpose(rescaled_image, (2, 0, 1)), axis=1)

    # loop through all the PNG files and save them
    for i in range(rescaled_image.shape[2]):
        # Convert numpy array to list
        array_list = np.rot90(rescaled_image[:, :, i], 1).tolist()
        # Convert list to a PIL image
        img = Image.fromarray(np.uint8(array_list))

        # Resize and crop
        img = resize_and_crop_center(img, 512, 512)

        # Save image as 3 channel RGB
        img = ImageOps.grayscale(img)
        img = img.convert('RGB')

        # Save the image
        img.save(os.path.join(outdir, f"im{i}.png"))


# inputdir = '/home/jabbar/results_project_tomo/data/Scapis'
inputdir = './data/training'

dir_i = 0
for filename in os.listdir(inputdir):
    # list all foldes within the input directory
    if os.path.isdir(os.path.join(inputdir, filename)):
        # Create the output directory

        outputdir_ct = os.path.join(
            './data/compiled', 'CT/' + str(dir_i).zfill(4))
        outputdir_tomo = os.path.join(
            './data/compiled', 'TOMO/' + str(dir_i).zfill(4))
        # Call the function to transform the CT scans
        print(os.path.join(inputdir, filename + '/CT/Thorax insp'))
        transform_ct(os.path.join(inputdir, filename +
                     '/CT/Thorax insp'), outputdir_ct)
        transform_tomosynthesis(os.path.join(inputdir, filename +
                                             '/DX/Tomosynthesis'), outputdir_tomo)
        dir_i += 1
