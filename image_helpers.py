from PIL import Image, ImageOps

WHITE = (255, 255, 255)

def sizeImage(image, width, height):
    size = (width, height)
    result = Image.new("RGB", size, WHITE) # white background
    img = image.copy()
    img.thumbnail(size)
    x_offset = (size[0] - img.width)//2
    y_offset = (size[1] - img.height)//2
    top_left = (x_offset, y_offset)
    result.paste(img, top_left)
    return result
	
def loadSizedImage(filename, size):
    #result = Image.new("RGB", size, WHITE) # white background
    img = Image.open(filename)
    #img.thumbnail(size)
    #x_offset = (size[0] - img.width)//2
    #y_offset = (size[1] - img.height)//2
    #top_left = (x_offset, y_offset)
    #result.paste(img, top_left)
    result = sizeImage(img, size[0], size[1])
    return result

def loadImage128x128(filename):
    return loadSizedImage(filename, (128, 128))

def processImage(img):
    # img = ImageOps.equalize(img)
    img = ImageOps.autocontrast(img)
    return img