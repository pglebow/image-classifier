__author__ = 'pglebow'

img_dir = "images/"
images = [img_dir+ f for f in os.listdir(img_dir)]
labels = ["check" if "check" in f.split('/')[-1] else "drivers_license" for f in images]

data = []
for image in images:
    img = img_to_matrix(image)
    img = flatten_image(img)
    data.append(img)

data = np.array(data)
data