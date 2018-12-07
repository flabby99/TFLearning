import inviwopy
import ivw.utils as inviwo_utils
from inviwopy.glm import ivec2

import os

def main(save_dir, image_name, pixel_dim, tf_location):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_loc = os.path.join(save_dir, image_name)

    inviwopy.app.waitForPool()
    network = inviwopy.app.network
    canvases = network.canvases
    for canvas in canvases:
        canvas.inputSize.dimensions.value = ivec2(pixel_dim, pixel_dim)

    tf = network.VolumeRaycaster.isotfComposite.transferFunction
    tf.load(tf_location)

    # Update the network
    inviwo_utils.update()

    canvas.snapshot(save_loc)
    inviwopy.app.closeInviwoApplication()
    return 0

if __name__ == "__main__":
    save_dir = os.path.join(
        os.path.expanduser('~'),    
        "TransferFunctionLearning",
        "GeneratedImages")
    save_name = "ivw_image.png"
    pixel_dim = 128
    tf_location = os.path.join(
        os.path.expanduser('~'),
        "TransferFunctionLearning",
        "GeneratedTFs",
        "gen_tf.itf")
    main(save_dir, save_name, pixel_dim, tf_location)