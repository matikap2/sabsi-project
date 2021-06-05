from PIL import Image
class ImageLoader:
    """!
    @brief Load image from given file and convert it into list of pixels(list of RGB colours)
    @param[in] path_to_image Path to image
    """
    def __init__(self, path_to_image : str):
        self._path_to_image = path_to_image
        self._rgb_data = list()
        self.load_image()

    """!
    @brief Load image from file and convert it into list of RGB pixels
    """
    def load_image(self):
        for pixel in list(Image.open(self._path_to_image).getdata()):
            self._rgb_data.append(list(pixel))
    """!
    @brief Print collected data
    """
    def print_rgb_data(self):
        print(self._rgb_data)