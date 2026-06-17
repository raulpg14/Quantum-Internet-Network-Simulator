from PIL import ImageTk, Image
import logging

logger = logging.getLogger(__name__)

def read_image(path, size):
    return ImageTk.PhotoImage(Image.open(path).resize(size, Image.ADAPTIVE))
