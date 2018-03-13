from PIL import Image

WHITE = (255, 255, 255)

def loadSizedImage(filename, size):
    result = Image.new("RGB", size, WHITE) # white background
    img = Image.open(filename)
    img.thumbnail(size)
    x_offset = (size[0] - img.width)//2
    y_offset = (size[1] - img.height)//2
    top_left = (x_offset, y_offset)
    result.paste(img, top_left)
    return result

