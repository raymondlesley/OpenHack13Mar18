CLASSES = [
    "axes",
    "boots",
    "carabiners",
    "crampons",
    "gloves",
    "hardshell_jackets",
    "harnesses",
    "helmets",
    "insulated_jackets",
    "pulleys",
    "rope",
    "tents"
]

def getClassIdx(classname):
    idx = CLASSES.index(classname)
    return idx

def getNumClasses():
    return len(CLASSES)
