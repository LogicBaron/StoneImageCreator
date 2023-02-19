from PIL import Image
from torchvision.transforms.functional import to_pil_image

def make_grid(images, rows, cols):
    c, w, h = images[0].size()
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(to_pil_image(image), box=(i%cols*w, i//cols*h))
    return grid