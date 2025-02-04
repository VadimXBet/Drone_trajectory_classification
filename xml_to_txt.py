import os
from xml.etree.ElementTree import parse

def parse_xml(XML_PATH):
    output_file = os.path.join(XML_PATH[:-4] + "_LABELS.txt")
    if os.path.exists(output_file):
        os.remove(output_file)
    tree = parse(XML_PATH)
    root = tree.getroot()
    metas = root.findall("track")
    for meta in metas:
        id = meta.attrib["id"]
        label = 'birds' if meta.attrib["label"] == 'other' else 'drones'

        object_metas = meta.findall("box")
        for bbox in object_metas:
            xtl = float(bbox.attrib["xtl"])
            ytl = float(bbox.attrib["ytl"])
            xbr = float(bbox.attrib["xbr"])
            ybr = float(bbox.attrib["ybr"])
            frame = int(bbox.attrib["frame"])

            f = open(output_file, "a")
            f.write(f"{frame+1},{id},{xtl},{ytl},{xbr-xtl},{ybr-ytl},{label},1,-1,-1,-1\n")
            f.close()
    
    os.remove(XML_PATH)

if __name__ == '__main__':
    path = 'test_data\gt'
    for file_name in os.listdir(path):
        # parse_xml(os.path.join(path, file))
        with open(os.path.join(path, file_name), 'r') as file: 
            data = file.read() 
            data = data.replace('birds', '0') 
            data = data.replace('drones', '1') 
        
        with open(os.path.join(path, file_name), 'w') as file: 
            file.write(data)

        print(f'File {file_name} is processed')
