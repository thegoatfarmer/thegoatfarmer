import pandas as pd
import numpy as np
import cv2
from PIL import Image
import base64
import requests
import json
import re
from collections import deque, Counter

def build_collage(images):
    images = [im for im in images if im is not None]
    if len(images) == 0:
        return np.zeros((512, 512, 3))
    best_ix = np.argmax([np.shape(im)[0] * np.shape(im)[1] for im in images])
    h, w, c = images[best_ix].shape
    ims = [[cv2.resize(im, (w, h)), (w, h)] for im in images]
    df = pd.DataFrame(ims, columns=['Image', 'Size'])
#     df.sort_values(by = ['Size'], key= lambda col: col.map(lambda x: max(x)), ascending=False, inplace=True)
#     best_shape = df.iloc[0]['Size']
#     df['Image'] = df.apply(lambda row: cv2.resize(row['Image'], best_shape), axis=1)
#     df['Size'] = (best_shape[0], best_shape[1], 3)
    
    p = Packer()
    blocks = [Block(i) for i in df['Size']]
    p.fit(blocks)
    df['Locations'] = [i.fit.location for i in blocks]
    
    c_w = max(df.apply(lambda x: x.Locations[0] + x.Size[0], axis=1).values)
    c_h = max(df.apply(lambda x: x.Locations[1] + x.Size[1], axis=1).values)
    
    collage = np.zeros((c_h, c_w, 3))
    
    for ind, row in df.iterrows():
        im_arr = np.array(row['Image'])
        h, w, _ = im_arr.shape
        loc_x, loc_y = row['Locations']
        collage[loc_y:loc_y+h, loc_x:loc_x+w, :] = im_arr
    
    return collage.astype(np.uint8)

def get_ocr(image):
    url = 'http://testapps.dtc.corp/COG/ADI/Api/Annotation/ToJpegs'
    imagingDocuments = []
    imagingDocuments.append({'Base64Bytes' : convert_cv2img_to_b64bytes(image),
                         'FileName' : 'test.jpg'})
    request = {'ImagingDocuments' : imagingDocuments}
    request_string = json.dumps(request)

    x = requests.post(url, data = request_string, headers={'Content-Type': "application/json"})
    result = json.loads(x.content)[0]

    return result['OcrWords']

def get_piece_number(img):
    ocr = get_ocr(img)
    if ocr is None:
        ocr = []
    first_row = []
    second_row = []
    top_row_counts = {}
    bottom_row_counts = {}

    for b in ocr:
        if re.match('^\d{3}[A-z]$', b['Value']):
            first_row.append(b)
        if re.match('^[A-Z]{3}$', b['Value']) and b['Value'].upper() != 'ABF':
            second_row.append(b)
    if len(first_row) != 0:
        top_row_counts = dict(Counter([i['Value'] for i in first_row]))
        top_val = max(top_row_counts, key=top_row_counts.get)
        print(f'Top Result: {top_val}, Found in {top_row_counts[top_val]/20*100:.2f}% of frames\n')
    if len(second_row) != 0:
        bottom_row_counts = dict(Counter([i['Value'] for i in second_row]))
        bottom_val = max(bottom_row_counts, key=bottom_row_counts.get)
        print(f'Bottom Result: {bottom_val}, Found in {bottom_row_counts[bottom_val]/20*100:.2f}% of frames\n')

    return top_row_counts, bottom_row_counts

def convert_cv2img_to_b64bytes(img):
    ext = ".jpg"
    retval, buffer_img= cv2.imencode(ext, img)
    return base64.b64encode(buffer_img).decode("utf-8")

class Packer:
    """
    Defines a packer object to be used on a list of blocks.
    """
    def __init__(self):
        self.root = None

    def fit(self, blocks):
        """
        Initiates the packing.
            blocks: A list of block objects with a 'size' proprety representing (w,h) as a tuple.
        """
        self.root = Node((0, 0), blocks[0].size)

        for block in blocks:
            some_node = self.find_node(self.root, block.size)
            if some_node is not None:
                block.fit = self.split_node(some_node, block.size)
            else:
                block.fit = self.grow_node(block.size)

        return None

    def find_node(self, some_node, size):
        if some_node.used:
            return self.find_node(some_node.right, size) or self.find_node(some_node.down, size)
        elif (size[0] <= some_node.size[0]) and (size[1] <= some_node.size[1]):
            return some_node
        else:
            return None

    def split_node(self, some_node, size):
        some_node.used = True
        some_node.down = Node((some_node.location[0], some_node.location[1] + size[1]),
                              (some_node.size[0], some_node.size[1] - size[1]))
        some_node.right = Node((some_node.location[0] + size[0], some_node.location[1]),
                               (some_node.size[0] - size[0], size[1]))
        return some_node

    def grow_node(self, size):
        can_go_down = size[0] <= self.root.size[0]
        can_go_right = size[1] <= self.root.size[1]

        should_go_down = can_go_down and (self.root.size[0] >= (self.root.size[1] + size[1]))
        should_go_right = can_go_right and (self.root.size[1] >= (self.root.size[0] + size[0]))

        if should_go_right:
            return self.grow_right(size)
        elif should_go_down:
            return self.grow_down(size)
        elif can_go_right:
            return self.grow_right(size)
        elif can_go_down:
            return self.grow_down(size)
        else:
            return None

    def grow_right(self, size):
        new_root = Node((0, 0), (self.root.size[0] + size[0], self.root.size[1]))
        new_root.used = True
        new_root.down = self.root
        new_root.right = Node((self.root.size[0], 0), (size[0], self.root.size[1]))

        self.root = new_root

        some_node = self.find_node(self.root, size)
        if some_node is not None:
            return self.split_node(some_node, size)
        else:
            return None

    def grow_down(self, size):
        new_root = Node((0, 0), (self.root.size[0], self.root.size[1] + size[1]))
        new_root.used = True
        new_root.down = Node((0, self.root.size[1]), (self.root.size[0], size[1]))
        new_root.right = self.root

        self.root = new_root

        some_node = self.find_node(self.root, size)
        if some_node is not None:
            return self.split_node(some_node, size)
        else:
            return None

class Block:
    """
    Defines an object Block with two properties.
        size: tuple representing the blocks size (w,h)
         fit: Stores a Node object for output.
    """
    def __init__(self, size):
        self.size = size
        self.fit = None

class Node:
    """
    Defines an object Node for use in the packer function.  Represents the space that a block is placed.
        used: Boolean to determine if a node has been used.
        down: A node located beneath the current node.
        right: A node located to the right of the current node.
        size: A tuple (w,h) representing the size of the node.
        location: A tuple representing the (x,y) coordinate of the top left of the node.
    """
    def __init__(self, location, size):
        self.used = False
        self.down = None
        self.right = None
        self.size = size
        self.location = location