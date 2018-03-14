from pathlib import Path
'''
p = Path('gear_images')
#[x for x in p.iterdir() if x.is_dir()]
for dir in p.iterdir():
    print(dir)
    if (dir.is_dir):
        for file in dir.iterdir():
            print(file)
'''

def loadDirs(directory):
    p = Path(directory)
    # print("p=", p)
    dirs = [p]

    for f in p.iterdir():
        # print(f)
        if (f.is_dir()):
            # print(f, "dir")t
            dirs.append(f)
            # dirs.append(loadDirs(f))

    return dirs

def loadImages(directory):
    dir = Path(directory)
    images = []
    for f in dir.iterdir():
        if (f.is_file()):
            images.append(f)
    return images

def loadImagesRecursive(directory, max_per_dir=9999):
    dirs = loadDirs(directory)
    # print(dirs)
    images = []
    
    for d in dirs:
        # print("looking in", d)
        if d.is_dir():
            num_this_dir = 0
            for f in d.iterdir():
                if (f.is_file()):
                    # print(f, "is a file")
                    images.append(f)
                    if (num_this_dir >= max_per_dir):
                        break
                    num_this_dir += 1
            # print("done with", d)

    return images

from pathlib import Path
from image_helpers import *

def processAllImages(input_dir, output_dir, process, max_per_dir=9999):
    all_images = loadImagesRecursive(input_dir, max_per_dir)

    for image in all_images:
        # print("from:", image.parent.name + '/' + image.name)
        img = loadImage128x128(image)
        # print("loaded:", img)
        new_img = process(img)
        #filename = output_dir + '/' + str(image_number) + '.jpg'
        dirname = output_dir + '/' + image.parent.name
        directory = Path(dirname)
        if (not directory.exists()):
            # print("... which doesn't exist")
            directory.mkdir(parents=True)
        filename = output_dir + '/' + image.parent.name + '/' + image.name
        #print("... to:", filename)
        try:
            new_img.save(filename)
        except:
            print("problem with", filename, "(didn't save)")

def loadImagesForSciKit(input_dir, process):
    all_images = loadImagesRecursive(input_dir)
    output_data = []

    for image in all_images:
        print("from:", image.parent.name + '/' + image.name)
        img = loadImage128x128(image)
        # print("loaded:", img)
        image_data = process(img)
        #filename = output_dir + '/' + str(image_number) + '.jpg'
        dirname = image.parent.name  # class
        print("Image class ", dirname)
