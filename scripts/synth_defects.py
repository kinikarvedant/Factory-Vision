import os, random, math
from PIL import Image, ImageDraw, ImageFilter

OUT = "dataset"
os.makedirs(f"{OUT}/images/train", exist_ok=True)
os.makedirs(f"{OUT}/images/val", exist_ok=True)
os.makedirs(f"{OUT}/labels/train", exist_ok=True)
os.makedirs(f"{OUT}/labels/val", exist_ok=True)

def make_base(size=(640,480), color=(200,200,200)):
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    pad = 40
    draw.rectangle([pad, pad, size[0]-pad, size[1]-pad], outline=(80,80,80), width=3)
    return img

def add_scratch(img):
    w,h = img.size
    draw = ImageDraw.Draw(img)
    x1,y1 = random.randint(60, w-60), random.randint(60, h-60)
    angle = random.uniform(-math.pi/3, math.pi/3)
    length = random.randint(50, 200)
    x2 = int(x1 + length*math.cos(angle))
    y2 = int(y1 + length*math.sin(angle))
    draw.line([x1,y1,x2,y2], fill=(10,10,10), width=random.randint(2,6))
    xmin, ymin = min(x1,x2)-5, min(y1,y2)-5
    xmax, ymax = max(x1,x2)+5, max(y1,y2)+5
    return (xmin, ymin, xmax, ymax), img

def add_spot(img):
    w,h = img.size
    draw = ImageDraw.Draw(img)
    rx, ry = random.randint(100, w-100), random.randint(100, h-100)
    r = random.randint(12,40)
    draw.ellipse([rx-r, ry-r, rx+r, ry+r], fill=(30,30,30))
    xmin, ymin, xmax, ymax = rx-r, ry-r, rx+r, ry+r
    return (xmin,ymin,xmax,ymax), img

def save_example(idx, mode="train"):
    base = make_base()
    if random.random() < 0.6:
        box, img = add_scratch(base)
        cls = 0
    else:
        box, img = add_spot(base)
        cls = 1
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.5)))
    fn_img = f"{OUT}/images/{mode}/{idx:05d}.jpg"
    fn_lbl = f"{OUT}/labels/{mode}/{idx:05d}.txt"
    img.save(fn_img)
    w,h = img.size
    xmin, ymin, xmax, ymax = box
    x_c = (xmin + xmax) / 2 / w
    y_c = (ymin + ymax) / 2 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    with open(fn_lbl, "w") as f:
        f.write(f"{cls} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

for i in range(200): save_example(i, "train")
for i in range(60): save_example(i, "val")
print("Generated dataset in dataset/images and dataset/labels")
