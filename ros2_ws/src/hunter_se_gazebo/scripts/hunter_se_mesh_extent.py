#!/usr/bin/env python3

import math
import os
import struct


MESH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "meshes",
)

FRONT_AXLE_X = 0.339638
REAR_AXLE_X = -0.208058
TRACK = 0.503404
WHEEL_VERTICAL_OFFSET = -0.16713
WHEEL_RADIUS = 0.136
BASE_LINK_Z = WHEEL_RADIUS - WHEEL_VERTICAL_OFFSET


def load_stl_vertices(path):
    with open(path, "rb") as stream:
        data = stream.read()

    if len(data) < 84:
        raise ValueError(f"STL too small: {path}")

    triangle_count = struct.unpack("<I", data[80:84])[0]
    expected_size = 84 + triangle_count * 50

    vertices = []
    if expected_size == len(data):
        offset = 84
        for _ in range(triangle_count):
            offset += 12
            for _ in range(3):
                vertices.append(struct.unpack("<3f", data[offset:offset + 12]))
                offset += 12
            offset += 2
        return vertices

    text = data.decode("utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex "):
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))
    return vertices


def bounds(points):
    mins = [min(p[i] for p in points) for i in range(3)]
    maxs = [max(p[i] for p in points) for i in range(3)]
    return mins, maxs


def matmul(a, b):
    return [
        [
            sum(a[row][k] * b[k][col] for k in range(3))
            for col in range(3)
        ]
        for row in range(3)
    ]


def rpy_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    return matmul(rz, matmul(ry, rx))


def transform(points, xyz, rpy):
    rot = rpy_matrix(*rpy)
    tx, ty, tz = xyz
    out = []
    for x, y, z in points:
        out.append((
            rot[0][0] * x + rot[0][1] * y + rot[0][2] * z + tx,
            rot[1][0] * x + rot[1][1] * y + rot[1][2] * z + ty,
            rot[2][0] * x + rot[2][1] * y + rot[2][2] * z + tz,
        ))
    return out


def load_mesh(name):
    return load_stl_vertices(os.path.join(MESH_DIR, name))


def print_size(label, mins, maxs):
    size = [maxs[i] - mins[i] for i in range(3)]
    print(
        f"{label}: "
        f"x={size[0]:.4f} m ({size[0] * 1000:.1f} mm), "
        f"y={size[1]:.4f} m ({size[1] * 1000:.1f} mm), "
        f"z={size[2]:.4f} m ({size[2] * 1000:.1f} mm)"
    )


def main():
    parts = []

    base_mesh = load_mesh("base_link.STL")
    parts.append(transform(base_mesh, (0.0, 0.0, BASE_LINK_Z), (0.0, 0.0, 0.0)))

    front_meshes = [
        ("fr_steer_left_link.STL",  FRONT_AXLE_X,  TRACK / 2,  (0.0, 0.0, 0.0)),
        ("fr_left_link.STL",        FRONT_AXLE_X,  TRACK / 2,  (1.5708, 0.0, 0.016976)),
        ("fr_steer_right_link.STL", FRONT_AXLE_X, -TRACK / 2,  (0.0, 0.0, 0.0)),
        ("fr_right_link.STL",       FRONT_AXLE_X, -TRACK / 2,  (-1.5708, 0.0, 0.0)),
    ]
    rear_meshes = [
        ("re_left_link.STL",  REAR_AXLE_X,  TRACK / 2,  (1.5708, 0.0, 0.0)),
        ("re_right_link.STL", REAR_AXLE_X, -TRACK / 2,  (-1.5708, 0.0, 0.0)),
    ]

    for mesh_name, x, y, rpy in front_meshes + rear_meshes:
        mesh = load_mesh(mesh_name)
        parts.append(transform(
            mesh,
            (x, y, BASE_LINK_Z + WHEEL_VERTICAL_OFFSET),
            rpy,
        ))

    merged = [point for part in parts for point in part]
    mins, maxs = bounds(merged)

    print("Hunter SE visual extents at zero steering")
    print(f"mesh_dir: {MESH_DIR}")
    print_size("overall", mins, maxs)
    print(
        f"min corner: x={mins[0]:.4f}, y={mins[1]:.4f}, z={mins[2]:.4f}\n"
        f"max corner: x={maxs[0]:.4f}, y={maxs[1]:.4f}, z={maxs[2]:.4f}"
    )


if __name__ == "__main__":
    main()
