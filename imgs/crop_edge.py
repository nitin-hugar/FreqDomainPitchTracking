from PIL import Image
from glob import glob
import os


OUTPUT_FOLDER = "CROPPED"
WHITE = (255, 255, 255, 255)


def check(px, w, h, rev_w: bool, rev_h: bool, rev_for: bool):
    wr = range(w - 1, -1, -1) if rev_w else range(w)
    hr = range(h - 1, -1, -1) if rev_h else range(h)
    if rev_for:
        for j in hr:
            for i in wr:
                if px[i, j] != WHITE:
                    return j
    else:
        for i in wr:
            for j in hr:
                if px[i, j] != WHITE:
                    return i


def extract_bool(v: int, cnt: int=3):
    for _ in range(cnt):
        yield bool(v & 1)
        v >>= 1


def crop_one(filename: str):
    out_filename = OUTPUT_FOLDER + "/" + filename
    if os.path.isfile(out_filename):
        print("%s exists, skipped" % out_filename)
        return
    img = Image.open(filename)
    w, h = img.size
    px = img.load()

    l, r, u, b = [check(px, w, h, *extract_bool(v)) for v in (0, 1, 4, 6)]

    print(filename, l, u, r, b)
    cropped = img.crop((l, u, r + 1, b + 1))
    cropped.save(out_filename)


if __name__ == "__main__":
    for path in glob("*"):
        if os.path.isdir(path) and path != OUTPUT_FOLDER:
            output_path = OUTPUT_FOLDER + "/" + path
            if not os.path.isdir(output_path):
                os.mkdir(output_path)
            for filename in glob("%s/*.png" % path):
                crop_one(filename)
    for filename in glob("*.png"):
        crop_one(filename=filename)