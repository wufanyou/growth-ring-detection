import xml.etree.ElementTree as ET
from typing import Optional
from xml.etree.ElementTree import Element

import numpy as np
from path import Path


class CVATImageAnnotation:
    def __init__(self, element: Element):
        self.element = element
        self.file = Path(element.get("name"))

    def add_point(
        self,
        label: str,
        point: Optional[list] = None,
        offset: Optional[list] = (0, 0),
        scale: float = 5.0,
    ) -> None:
        """
        :param offset:
        :param label: label for point
        :param point: point coordinate
        :return:
        """
        try:
            length = len(point)
        except:
            length = 0

        if length == 2:
            if offset:
                point[0] += offset[0]
                point[1] += offset[1]
            point = [x / scale for x in point]
            point = ET.Element(
                "points",
                label=label,
                occluded="0",
                source="manual",
                points=f"{point[0]:.2f},{point[1]:.2f}",
                z_order="0",
            )
            self.element.append(point)

    def add_tag(self, label: str = "Species", attribute: Optional[dict] = None) -> None:
        """
        :param label: label for tag
        :param attribute: attributes as dicts
        :return: None
        """
        tag = ET.Element("tag", label=label, source="manual")
        if attribute is not None:
            for k, v in attribute.items():
                attr = ET.Element("attribute", name=str(k))
                attr.text = str(v)
                tag.append(attr)
        self.element.append(tag)

    def add_polygon(self, label, points: list) -> None:
        """
        :param label: label for polygon
        :param points: contour
        :return: None
        """
        points = ";".join([f"{p[0]:.2f},{p[1]:.2f}" for p in points])

        polygon = ET.Element(
            "polygon",
            label=label,
            occluded="0",
            source="manual",
            points=points,
            z_order="0",
        )

        self.element.append(polygon)

    def add_polyline(self, label, points: list) -> None:
        """
        :param label: label for polyline
        :param points: points
        :return: None
        """
        points = ";".join([f"{p[0]:.2f},{p[1]:.2f}" for p in points])

        polyline = ET.Element(
            "polyline",
            label=label,
            occluded="0",
            source="manual",
            points=points,
            z_order="0",
        )

        self.element.append(polyline)

    def get_dict(self) -> dict:
        """
        :return: dict of annotation
        """
        o = {}
        o["filename"] = str(self.file.name)
        for c in self.element:
            if c.tag == "points":
                o[c.get("label")] = np.array(
                    [float(x) for x in c.get("points").split(",")]
                )
            elif c.tag == "tag":
                label = c.get("label")
                if label == "Species":
                    o["species"] = c[0].text
                elif label == "Side:1" or label == "Side:0":
                    o["side"] = int(label.split(":")[1])
                elif label == "Dry" or label == "Wet":
                    o["moisture"] = label
                elif label == "Clean" or label == "Rough":
                    o["surface"] = label
                elif label == "ID":
                    o["ID"] = int(c[0].text)
            elif c.tag == "polyline":
                p = c.get("points")
                p = [np.array([float(y) for y in x.split(",")]) for x in p.split(";")]
                if c.get("label") in o:
                    o[c.get("label")].append(p)
                else:
                    o[c.get("label")] = [p]
        return o
