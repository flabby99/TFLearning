from random import random
import os

def random_rgba():
    return random(), random(), random(), random()


def int_to_float_density(density_int, max_int=256):
    if density_int is 0:
        return density_int
    scale = float(1 / max_int)
    return float(density_int * scale)

def generate_rgba_tf(max_int=256):
    tf = [
            [int_to_float_density(i, max_int), random_rgba()] 
            for i in range(max_int + 1)
         ]
    return tf

#rgba_val is [content, (rgba tf_val)]
def make_one_tf_point(rgba_val):
    rgba = rgba_val[1]
    content_string = "<pos content=\"{}\" />".format(rgba_val[0])
    rgba_string = "<rgba x=\"{}\" y=\"{}\" z=\"{}\" w=\"{}\" />".format(
        rgba[0], rgba[1], rgba[2], rgba[3]
    )
    return "\n".join((
        "<Point>",
        content_string,
        rgba_string,
        "</Point>"
    ))

def save_xml_tf(tf, out_location):
    xml_header = "\n".join((
        '<?xml version="1.0" ?>',
        '<InviwoWorkspace version="2">',
        '<maskMin content="0" />',
        '<maskMax content="1" />',
        '<type content="0" />'
    ))
    points = "\n".join(
        [make_one_tf_point(i) for i in tf]
    )
    points_string = '\n'.join((
        xml_header,
        "<Points>",
        points,
        "</Points>",
        "</InviwoWorkspace>"
    ))
    file = open(out_location, 'w')
    file.write(points_string)
    file.close()

def main():
    tf_base_dir = "GeneratedTFs"
    tf_name = "gen_tf.itf"
    out_location = os.path.join(tf_base_dir, tf_name)
    save_xml_tf(generate_rgba_tf(), out_location)
    return 0

if __name__ == "__main__":
    main()

