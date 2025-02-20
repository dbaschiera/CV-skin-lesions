import cv2

def remove_hair(image):
    """
    Applies hair removal using Black Top-Hat filtering and inpainting.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define structuring element (kernel) for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))  # Adjust for hair thickness

    # Apply morphological closing (X · Y)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # Closing is dilation followed by erosion
    # It helps to close small holes in the foreground objects (black hair pixels)

    # Compute Black Top-Hat transformation: B_TH = (X · Y) - X
    black_tophat = closed - gray

    # Threshold to create a binary mask of hair pixels
    _, mask = cv2.threshold(black_tophat, 10, 255, cv2.THRESH_BINARY)
    # Pixels that are white in closed but black in gray are likely hair pixels

    # Inpaint the detected hair pixels using neighboring pixels
    inpainted = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

    return inpainted


def resize_image(image, target_size=256, interpolation=cv2.INTER_LINEAR):
    """
    Resizes the image directly to target_size x target_size without keeping aspect ratio.
    No padding or cropping is applied.
    """
    return cv2.resize(image, (target_size, target_size), interpolation=interpolation)


# Target image size
target_size = 256

# Target image
image_path = 'my_skin_lesions/chest1.jpg'
write_path = 'my_normalized_skin_lesions/chest1.jpg'

# Read image
image = cv2.imread(image_path)

# Step 1: Apply hair removal on the original image
hair_removed_image = remove_hair(image)

# Step 2: Resize the hair-free image and mask to 256×256
resized_hair_removed = resize_image(hair_removed_image, target_size, interpolation=cv2.INTER_LINEAR)

cv2.imwrite(write_path, resized_hair_removed)  # 256x256 resized

