'''
Skript to change the URDF file of the goal object.
'''

import xml.etree.ElementTree as ET
import re

# settings
input_file = 'assets/objects/goals/goal_red.urdf'
output_file = 'assets/objects/goals/goal_red.urdf'
x_dez = 0.16 # defines inner rectangle
y_dez = 0.16
border_width = 0.001 # defines line width of rectangle


# calculation
x_w_g = x_dez
y_w_g = y_dez + 2*border_width

x1 = 0
y1 = (y_w_g-border_width)/2

x2 = 0
y2 = -(y_w_g-border_width)/2

x3 = -(x_w_g+border_width)/2
y3 = 0

x4 = (x_w_g+border_width)/2
y4 = 0

dimensions = {
    'x_length': x_w_g,
    'y_length': y_w_g,
    'border': border_width,
    'positions': {
        'top': {'x': x1, 'y': y1, 'z': 0},
        'bottom': {'x': x2, 'y': y2, 'z': 0},
        'left': {'x': x3, 'y': y3, 'z': 0},
        'right': {'x': x4, 'y': y4, 'z': 0}
    }
}
print(dimensions)

# Parse the URDF file
tree = ET.parse(input_file)
root = tree.getroot()

# Update box dimensions for each border
for link in root.findall('.//link'):
    link_name = link.get('name')
    box = link.find('.//geometry/box')
    
    if box is not None:
        if link_name in ['top_border', 'bottom_border']:
            # Update horizontal borders (x_length)
            box.set('size', f"{dimensions['x_length']} {dimensions['border']} {dimensions['border']}")
        elif link_name in ['left_border', 'right_border']:
            # Update vertical borders (y_length)
            box.set('size', f"{dimensions['border']} {dimensions['y_length']} {dimensions['border']}")

# Update joint positions
for joint in root.findall('.//joint'):
    joint_name = joint.get('name')
    origin = joint.find('origin')
    
    if origin is not None:
        if 'top_border_joint' in joint_name:
            origin.set('xyz', f"{dimensions['positions']['top']['x']} {dimensions['positions']['top']['y']} {dimensions['positions']['top']['z']}")
        elif 'bottom_border_joint' in joint_name:
            origin.set('xyz', f"{dimensions['positions']['bottom']['x']} {dimensions['positions']['bottom']['y']} {dimensions['positions']['bottom']['z']}")
        elif 'left_border_joint' in joint_name:
            origin.set('xyz', f"{dimensions['positions']['left']['x']} {dimensions['positions']['left']['y']} {dimensions['positions']['left']['z']}")
        elif 'right_border_joint' in joint_name:
            origin.set('xyz', f"{dimensions['positions']['right']['x']} {dimensions['positions']['right']['y']} {dimensions['positions']['right']['z']}")

# Write the modified URDF to a new file

# Convert the tree to a string with proper indentation
xml_str = ET.tostring(root, encoding='unicode')

# Add proper indentation
from xml.dom import minidom
xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="  ")

# Remove extra blank lines that minidom adds
xml_pretty = '\n'.join([line for line in xml_pretty.split('\n') if line.strip()])

# Write to file with XML declaration
with open(output_file, 'w') as f:
    f.write(xml_pretty)