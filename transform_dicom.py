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
    filenames = os.listdir(inputdir)
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split("Image ")[1].split()[0]))
    for filename in sorted_filenames:
        if filename.endswith(".dcm"):
            ds = pydicom.read_file(os.path.join(inputdir, filename), force=True)
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
    RefDs = pydicom.dcmread(os.path.join(inputdir, os.listdir(inputdir)[0]), force=True)

    # Load dimensions based on the number of rows, columns, and slices
    ConstPixelDims = (int(RefDs.Rows), int(
        RefDs.Columns), len(os.listdir(inputdir)))

    # Load spacing values
    # ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(
    #    RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # Create a 3D array to hold the pixel data
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    filenames = os.listdir(inputdir)
    sorted_filenames = sorted(filenames, key=lambda x: int(x.split("Image ")[1].split()[0]))
    for filename in sorted_filenames:
        if filename.endswith(".dcm"):
            # read the file
            ds = pydicom.read_file(os.path.join(inputdir, filename), force=True)
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


inputdir = '/home/jabbar/results_project_tomo/data/Scapis'
# inputdir = './data/training'

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

        # check how many files are in the tomosynthesis folder
        tomo_inputdir = os.path.join(inputdir, filename + '/DX/Tomosynthesis')
        ct_inputdir = os.path.join(inputdir, filename + '/CT/Thorax insp')
        print(os.path.join(inputdir, filename), len(os.listdir(tomo_inputdir)))
        if len(os.listdir(tomo_inputdir)) == 65 and len(os.listdir(ct_inputdir)) > 503:
            print("use")
            try:
                transform_ct(ct_inputdir, outputdir_ct)
                transform_tomosynthesis(tomo_inputdir, outputdir_tomo)
            except:
                print("error")
            else:
                dir_i += 1

