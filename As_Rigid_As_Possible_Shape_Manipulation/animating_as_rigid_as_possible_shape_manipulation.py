"""
As-Rigid-As-Possible Shape Manipulation

based on paper by Igarashi, Moscovich, Hughes, ACM SIGGRAPH 2005

http://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/rigid.pdf
http://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/takeo_jgt09_arapFlattening.pdf

This file is released under GNU Affero General Public Licence v3:
https://opensource.org/licenses/agpl-3.0

If you intend to use this source as a part of your own software or SaaS product, you are required to release full source
code of your product to your users
"""

import math

import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import shapely.geometry
import shapely.ops
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint


# is_debug=True
is_debug=False

def load_image(file_name):
    image = cv2.imread(file_name)
    return image


def adjust_boundary_segment_length(boundary, spacing):
    """
    Adjust boundary to contain segments of at most spacing size by splitting segments linearly
    :param boundary: list of points
    :param spacing: maximal distance between boundary points
    :return: boundary with required spacing
    """
    result = []
    i = 1;
    while i < len(boundary):
        x0, y0 = boundary[i - 1]
        x1, y1 = boundary[i]
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        result.append([x0, y0])
        if length > spacing:
            segments = int(round(length / spacing))
            for j in range(1, segments):
                n_x = x0 + dx * float(j) / float(segments)
                n_y = y0 + dy * float(j) / float(segments)
                result.append([n_x, n_y])
        result.append([x1, y1])
        i += 1

    return result


def triangulate(boundary, max_segments):
    """
    Triangulates boundary via equidistant points; subsequent rows shifted by half spacing
    :param boundary: list of boundary points forming a closed shape
    :param max_segments: maximal number of triangulated segments in either dimension
    :return: regularly-spaced, non-convex triangulation of the boundary
    """
    # polygon for containment testing
    area = Polygon(boundary)
    bounds = area.bounds

    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y

    dimension = max(width, height)
    spacing = dimension / max_segments

    boundary = adjust_boundary_segment_length(boundary, spacing)
    area = Polygon(boundary)

    points = [p for p in boundary]
    y = min_y + spacing / 2
    is_odd = True
    while y < max_y:
        x = min_x + spacing / 2
        if is_odd:
            x += spacing / 2
        while x < max_x:
            point = Point((x, y))
            if area.contains(point):
                points.append([x, y])
            x += spacing
        y += spacing
        is_odd = not is_odd

    triangulation_points = MultiPoint(points)

    triangulation = shapely.ops.triangulate(triangulation_points)

    # prune convex-shape triangles outside original boundaries

    result = []
    for polygon in triangulation:
        # intersection = area.intersection(polygon)
        # if intersection.area > 0.1 * polygon.area:
        coordinates = list(polygon.exterior.coords)
        center_x = coordinates[0][0] + coordinates[1][0] + coordinates[2][0]
        center_y = coordinates[0][1] + coordinates[1][1] + coordinates[2][1]
        center_x /= 3.
        center_y /= 3.
        center_point = Point([center_x, center_y])
        if area.contains(center_point):
            result.append(polygon)

    return result

def remove_background(image):
    """
    TODO: Taken from an OpenCV tutorial, is not used due to insufficient performance on real-world images
    TODO: Removes image background using watershed algorithm
    TODO: Works really bad...
    TODO: Forget this method
    :param image:
    :return:
    """
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    cv2.imshow("Input", image)
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Thresh", thresh)
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then apply the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.drawContours(image, cnts, -1, (200, 200, 200), 3)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)

        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

def visualize_triangulation(image, triangulation, file_name):
    """
    Visualizes triangulation inside existing image into a new image stored in given filename
    :param image: image
    :param triangulation: triangulation
    :param file_name: file name
    """
    drawing = image.copy()
    for polygon in triangulation:
        coordinates = list(polygon.exterior.coords)
        pixels = [(int(x), int(y)) for x, y in coordinates]
        for i in range(3):
            cv2.line(drawing, pixels[i], pixels[i + 1], (70, 120, 200), 1)
    cv2.imwrite(file_name, drawing)

def visualize_edges(image, points, edges, file_name, handles=None):
    """
    Visualizes edges inside existing image into a new image stored in given filename
    :param image: image
    :param points: mesh points
    :param edges: mesh edges
    :param file_name: file name
    :param handles: displays handles as circles if True
    """
    drawing = image.copy()
    for edge in edges:
        v_i, v_j = edge
        p_i = (int(points[v_i, 0]), int(points[v_i, 1]))
        p_j = (int(points[v_j, 0]), int(points[v_j, 1]))
        for i in range(3):
            cv2.line(drawing, p_i, p_j, (70, 120, 200), 1)

    if handles is not None:
        for handle in handles:
            p = (int(handle[0]), int(handle[1]))
            cv2.circle(drawing, p, 10, (0, 210, 210), -1)
    cv2.imwrite(file_name, drawing)

def visualize_contour(image, contour, file_name):
    """
    Visualizes contour inside existing image into a new image stored in given filename
    :param image: image
    :param contour: contour as list of [x, y] coordinates
    :param file_name: file name
    """
    drawing = image.copy()
    p_prev = contour[len(contour) - 1]
    p_prev = (int(p_prev[0]), int(p_prev[1]))

    for i in range(len(contour)):
        p = contour[i]
        p = (int(p[0]), int(p[1]))
        for i in range(3):
            cv2.line(drawing, p_prev, p, (70, 120, 200), 1)
            p_prev = p

    cv2.imwrite(file_name, drawing)

def get_edges_and_neighbors(triangulation):
    """
    Prepares meta information needed for pupper
    :param triangulation: triangulation
    :return: list of points, lookup list of point-to-index, set of edges and dictionary of edge neighbors for each edge
    """
    points = set()
    point_index = {}
    index_point = {}
    edges = set()
    edge_neighbors = {}
    i = 0
    for polygon in triangulation:
        coordinates = list(polygon.exterior.coords)
        p0_x, p0_y = coordinates[0]
        p1_x, p1_y = coordinates[1]
        p2_x, p2_y = coordinates[2]

        # index points
        triangle_points = [(p0_x, p0_y), (p1_x, p1_y), (p2_x, p2_y)]
        for point in triangle_points:
            if point not in points:
                points.add(point)
                point_index[point] = i
                index_point[i] = point
                i += 1

    for polygon in triangulation:
        coordinates = list(polygon.exterior.coords)
        p0_x, p0_y = coordinates[0]
        p1_x, p1_y = coordinates[1]
        p2_x, p2_y = coordinates[2]
        i_0 = point_index[(p0_x, p0_y)]
        i_1 = point_index[(p1_x, p1_y)]
        i_2 = point_index[(p2_x, p2_y)]

        # gather edges
        triangle_edges = [(i_0, i_1), (i_1, i_2), (i_0, i_2)]
        for i_a, i_b in triangle_edges:
            i = min(i_a, i_b)
            j = max(i_a, i_b)

            if (i, j) not in edges:
                edges.add((i, j))
                neighbors = set()
                # add all triangle vertices as edge neighbors
                neighbors.add(i_0)
                neighbors.add(i_1)
                neighbors.add(i_2)
                edge_neighbors[(i, j)] = neighbors
            else:
                neighbors = edge_neighbors[(i, j)]
                # should add at most one vertex
                neighbors.add(i_0)
                neighbors.add(i_1)
                neighbors.add(i_2)

    points = []
    for i in range(len(index_point)):
        points.append(index_point[i])

    return points, point_index, edges, edge_neighbors

def test_edge_neighbors(edges, edge_neighbors):
    """
    Tests edge neighbors for expected structure and coverage
    :param edges: edges
    :param edge_neighbors: dictionary of edge neighbors
    """
    for edge in edges:
        neighbors = edge_neighbors[edge]
        if len(neighbors) < 3 or len(neighbors) > 4:
            print "Incorrect number of neighbors, e:", edge, "n:", neighbors


def get_barycentric_coordinates(triangulation, point, point_index):
    """
    Computes barycentric coordinate of a point v as o_1 * v1_x + o2 * v2_x + o_3 * v3_x inside first triangle containing it
    :param triangulation: triangulation
    :param point: point coordinates
    :param point_index: index of point coordinates in list of points
    :return: [point indices], [barycentric coordinates]
    """
    # find a triangle containing point
    for polygon in triangulation:
        coordinates = list(polygon.exterior.coords)
        area = Polygon(coordinates)
        p = Point(point)
        if area.contains(p):
            points = [point_index[coordinates[0]], point_index[coordinates[1]], point_index[coordinates[2]]]
            barycentric = [0, 0, 0]
            dy23 = coordinates[1][1] - coordinates[2][1]
            dy31 = coordinates[2][1] - coordinates[0][1]
            dx32 = coordinates[2][0] - coordinates[1][0]
            dx13 = coordinates[0][0] - coordinates[2][0]
            dx3 = point[0] - coordinates[2][0]
            dy3 = point[1] - coordinates[2][1]

            omega_1 = (dy23 * dx3 + dx32 * dy3) / (dy23 * dx13 - dx32 * dy31)
            omega_2 = (dy31 * dx3 + dx13 * dy3) / (dy23 * dx13 - dx32 * dy31)
            omega_3 = 1 - omega_1 - omega_2

            barycentric = [omega_1, omega_2, omega_3]
            return points, barycentric
    return None, None

def registration(points, point_index, edges, edge_neighbors, triangulation):
    """
    Computes matrices for the ARAP registration phase
    :param points: list of points
    :param point_index: dictionary of point-to-index relations
    :param edges: set of edges
    :param edge_neighbors: dictionary of neighbors for each edge
    :param triangulation: triangulation
    :return: matrix factors for 2nd computation phase
    """
    # first compute upper half of A_1, called L_1 for similarity transformation
    # this produces scaled version of what we want
    # then computer upper half of A_2, called C_1 for scale adjustment
    L_1 = np.zeros((2 * len(edges), 2 * len(points)))
    L_2 = np.zeros((len(edges), len(points)))
    G_ks = {}
    T_ks = {}
    row_1 = 0
    row_2 = 0
    for k, edge in enumerate(edges):

        neighbors = edge_neighbors[edge]
        dimension = len(neighbors)
        v_i, v_j = edge
        v_l = None
        v_r = None
        vertices = [v_i, v_j]
        for v in neighbors:
            if v != v_i and v != v_j:
                v_l = v
                vertices.append(v_l)
                break
        if dimension == 4:
            for v in neighbors:
                if v != v_i and v != v_j and v != v_l:
                    v_r = v
                    vertices.append(v_r)
                    break

        # edge vertices
        p_i = points[v_i]
        p_j = points[v_j]
        # edge neighbor vertices (other vertices of edge's at most two triangles)
        p_l = points[v_l]
        p_r = points[v_r] if dimension == 4 else None
        # edge
        e_k = np.subtract(p_j, p_i)
        # G_k is 2 * N x 2 matrix
        G_k = np.zeros((2 * dimension, 2))
        # v_i
        G_k[0, 0] =  p_i[0]
        G_k[0, 1] =  p_i[1]
        G_k[1, 0] =  p_i[1]
        G_k[1, 1] = -p_i[0]
        # v_j
        G_k[2, 0] =  p_j[0]
        G_k[2, 1] =  p_j[1]
        G_k[3, 0] =  p_j[1]
        G_k[3, 1] = -p_j[0]
        # v_l
        G_k[4, 0] =  p_l[0]
        G_k[4, 1] =  p_l[1]
        G_k[5, 0] =  p_l[1]
        G_k[5, 1] = -p_l[0]
        if p_r is not None:
            # v_r
            G_k[6, 0] =  p_r[0]
            G_k[6, 1] =  p_r[1]
            G_k[7, 0] =  p_r[1]
            G_k[7, 1] = -p_r[0]

        G_ks[(v_i, v_j)] = G_k

        # T_k = (G_k^T * G_k)^-1 * G_k^T
        G_kTG_k = np.dot(G_k.T, G_k)
        G_kTG_k_inv = np.linalg.inv(G_kTG_k)
        T_k = np.dot(G_kTG_k_inv, G_k.T)

        T_ks[(v_i, v_j)] = T_k

        H_v = np.zeros((2, 2 * dimension))
        H_v[0, 0] = -1
        H_v[1, 1] = -1
        H_v[0, 2] =  1
        H_v[1, 3] =  1

        H_e = np.zeros((2, 2))
        H_e[0, 0] =  e_k[0]
        H_e[0, 1] =  e_k[1]
        H_e[1, 0] =  e_k[1]
        H_e[1, 1] = -e_k[0]

        # H = [-1  0 1 0 0 0 0 0] - [e_kx  e_ky] * T_k
        #     [ 0 -1 0 1 0 0 0 0]   [e_ky -e_kx]
        H = H_v - np.dot(H_e, T_k)

        # registration, filling in H values to L_1
        for i, v in enumerate(vertices):
            L_1[row_1, v * 2] = H[0, i * 2]
            L_1[row_1, v * 2 + 1] = H[0, i * 2 + 1]
            L_1[row_1 + 1, v * 2] = H[1, i * 2]
            L_1[row_1 + 1, v * 2 + 1] = H[1, i * 2 + 1]

        row_1 += 2

        L_2[row_2, v_i] = -1
        L_2[row_2, v_j] =  1

        row_2 += 1

    A_1_factor_1 = np.dot(L_1.T, L_1)
    A_2_factor_1 = np.dot(L_2.T, L_2)

    # Fill in C_1 & C_2
    C_1 = np.zeros((2 * len(handles), 2 * len(points)))
    C_2 = np.zeros((len(handles), len(points)))
    b = np.zeros((2 * (len(edges) + len(handles)), 1))
    row_1 = 0
    row_2 = 0
    for k, handle in enumerate(handles):
        p_i, p_b = get_barycentric_coordinates(triangulation, handle, point_index)
        if p_i == None:
            continue

        for i in range(3):
            ix = p_i[i]
            bc = p_b[i]
            C_1[row_1, ix * 2] = bc * weight
            C_1[row_1 + 1, ix * 2 + 1] = bc * weight
            C_2[row_2, ix] = bc * weight
        b[2 * len(edges) + row_1] = weight * handle[0]
        b[2 * len(edges) + row_1 + 1] = weight * handle[1]

        row_1 += 2
        row_2 += 1

    A_1_factor_2 = np.dot(C_1.T, C_1)
    A_2_factor_2 = np.dot(C_2.T, C_2)

    A_1 = np.vstack((L_1, C_1))
    A_2 = np.vstack((L_2, C_2))

    factors = {}
    factors["A_1"] = A_1
    factors["A_2"] = A_2
    factors["T_ks"] = T_ks
    factors["G_ks"] = G_ks
    factors["L_1"] = L_1
    factors["L_2"] = L_2
    factors["A_1_factor1"] = A_1_factor_1
    factors["A_1_factor2"] = A_1_factor_2
    factors["A_2_factor1"] = A_2_factor_1
    factors["A_2_factor2"] = A_2_factor_2

    return factors


def compute_new_positions(points, edges, edge_neighbors, handles, weight, factors, image=None):
    """
    Computes new mesh deformation for given handle positions
    :param points: list of points
    :param edges: set of edges
    :param edge_neighbors: dictionary of neighbors of each edge
    :param handles: list of handle positions
    :param weight: weight of a handle
    :param factors: matrix factors from registration phase
    :param image: image
    :return: list of new mesh vertex positions
    """
    # Phase 1: Similarity Transform
    b_1 = np.zeros((2 * (len(edges) + len(handles)), 1))
    row_1 = 0
    for k, handle in enumerate(handles):
        for i in range(3):
            b_1[2 * len(edges) + row_1] = weight * handle[0]
            b_1[2 * len(edges) + row_1 + 1] = weight * handle[1]

        row_1 += 2

    A_1 = factors["A_1"]

    if is_debug:
        dump_matrix_as_csv(A_1, "A_1", "A_1.csv")

    LHS = np.dot(A_1.T, A_1)
    RHS = np.dot(A_1.T, b_1)

    moved_vertices = np.linalg.solve(LHS, RHS)
    moved_vertices = array_to_coordinates(moved_vertices)

    if is_debug:
        check_still_state_closeness(points, moved_vertices, 1E-5)
        if image is not None:
            original_points = np.array([[px, py] for px, py in points])
            visualize_edges(image, original_points, edges, "triangulation-phase0.png")
            visualize_edges(image, moved_vertices, edges, "triangulation-phase1.png")

    # Phase 2: Scale adjustment
    A_2 = factors["A_2"]

    if is_debug:
        dump_matrix_as_csv(A_2, "A_2", "A_2.csv")

    T_ks = factors["T_ks"]
    T2_ks = {}

    b2_x = np.zeros((len(edges) + len(handles), 1))
    b2_y = np.zeros_like(b2_x)

    for k, edge in enumerate(edges):

        neighbors = edge_neighbors[edge]
        dimension = len(neighbors)
        v_i, v_j = edge
        v_l = None
        v_r = None
        vertices = [v_i, v_j]
        for v in neighbors:
            if v != v_i and v != v_j:
                v_l = v
                vertices.append(v_l)
                break
        if dimension == 4:
            for v in neighbors:
                if v != v_i and v != v_j and v != v_l:
                    v_r = v
                    vertices.append(v_r)
                    break

        # take newly computed edge vertices and neighbors positions

        # edge vertices
        p_i = moved_vertices[v_i]
        p_j = moved_vertices[v_j]
        # edge neighbor vertices (other vertices of edge's at most two triangles)
        p_l = moved_vertices[v_l]
        p_r = moved_vertices[v_r] if dimension == 4 else None
        # edge
        e_k = np.subtract(p_j, p_i)

        vs = np.zeros((2 * dimension, 1))
        vs[0, 0] = p_i[0]
        vs[1, 0] = p_i[1]
        vs[2, 0] = p_j[0]
        vs[3, 0] = p_j[1]
        vs[4, 0] = p_l[0]
        vs[5, 0] = p_l[1]
        if dimension == 4:
            vs[6, 0] = p_r[0]
            vs[7, 0] = p_r[1]

        T_k = T_ks[(v_i, v_j)]

        rotations = np.dot(T_k, vs)

        c_k = rotations[0, 0]
        s_k = rotations[1, 0]

        norm = c_k * c_k + s_k * s_k
        T2_k = np.zeros((2, 2))

        T2_k[0, 0] = c_k / norm
        T2_k[0, 1] = s_k / norm
        T2_k[1, 0] = -s_k / norm
        T2_k[1, 1] = c_k / norm

        T2_ks[(v_i, v_j)] = T2_k

        e = np.zeros((2, 1))
        e[0, 0] = e_k[0]
        e[1, 0] = e_k[1]

        re = np.dot(T2_k, e)

        b2_x[k, 0] = re[0, 0]
        b2_y[k, 0] = re[1, 0]

    # set handle weights
    for k, handle in enumerate(handles):
        b2_x[len(edges) + k] = weight * handle[0]
        b2_y[len(edges) + k] = weight * handle[1]

    # solve systems to find new x and y coordinates
    LHS = np.dot(A_2.T, A_2)
    RHS = np.dot(A_2.T, b2_x)
    vertices_x = np.linalg.solve(LHS, RHS)
    RHS = np.dot(A_2.T, b2_y)
    vertices_y = np.linalg.solve(LHS, RHS)
    vertices = np.zeros((len(moved_vertices), 2))
    vertices[:, 0] = vertices_x[:, 0]
    vertices[:, 1] = vertices_y[:, 0]

    if is_debug:
        check_still_state_closeness(points, vertices, 1E-5)
        if image is not None:
            visualize_edges(image, vertices, edges, "triangulation-phase2.png")

    return vertices


def dump_matrix_as_csv(matrix, header=None, file_name=None):
    """
    Stores matrix as a CSV file for debugging reasons
    :param matrix: matrix
    :param header: first line of file
    :param file_name: file name
    """
    if file_name is None:
        if header is not None:
            print header
        height, width = matrix.shape
        for y in range(height):
            for x in range(width):
                print matrix[y, x],",",
            print
    else:
        with open(file_name, 'w') as f:
            if header is not None:
                f.write(header + "\n")
            height, width = matrix.shape
            for y in range(height):
                for x in range(width):
                    f.write(str(matrix[y, x]))
                    f.write(",")
                f.write("\n")

def array_to_coordinates(array):
    """
    Converts 1D array with consecutive X, Y pairs to a 2D array
    :param array:
    :return:
    """
    v_x = array[::2]
    v_y = array[1::2]

    v = np.zeros((len(v_x), 2))
    v[:, 0] = v_x[:, 0]
    v[:, 1] = v_y[:, 0]

    return v

def check_still_state_closeness(points, vertices, epsilon):
    """
    Test for mesh keeping the same shape when no handle changes
    :param points: list of initial mesh vertices
    :param vertices: list of newly computed mesh vertices
    :param epsilon: epsilon for L1 comparison
    """
    delta = np.subtract(points, vertices)
    max_delta = np.max(delta)
    if max_delta > epsilon:
        print "coordinates deviate! (", max_delta, " > ", epsilon, ")"

def get_indexed_triangles(triangulation, point_index):
    """
    Returns indexed triangles as tuples of (i1, i2, i3) where i is index of vertex v in point list
    :param triangulation: triangulation
    :param point_index: dictionary of point-to-index relations
    :return: triangles in indexed format
    """
    triangles = []
    for polygon in triangulation:
        coordinates = list(polygon.exterior.coords)
        i_0 = point_index[(coordinates[0][0], coordinates[0][1])]
        i_1 = point_index[(coordinates[1][0], coordinates[1][1])]
        i_2 = point_index[(coordinates[2][0], coordinates[2][1])]
        triangles.append((i_0, i_1, i_2))
    return triangles

def visualize_image(texture, texture_coordinates, vertices, triangles, file_name):
    """
    Visualizes warped image
    :param texture: initial image as texture
    :param texture_coordinates: initial mesh coordinates as texture coordinates
    :param vertices: list of mesh vertices
    :param triangles: list of indexed mesh triangles
    :param file_name: file name
    :return: resulting image
    """
    # result = np.zeros_like(texture)
    result = np.zeros((texture.shape[0] + 50, texture.shape[1] + 50, texture.shape[2]))

    for triangle in triangles:

        i_0, i_1, i_2 = triangle

        t_0 = texture_coordinates[i_0]
        t_1 = texture_coordinates[i_1]
        t_2 = texture_coordinates[i_2]
        source = np.float32([[t_0, t_1, t_2]])

        v_0 = vertices[i_0]
        v_1 = vertices[i_1]
        v_2 = vertices[i_2]
        target = np.float32([[v_0, v_1, v_2]])


        source_boundary = cv2.boundingRect(source)
        target_boundary = cv2.boundingRect(target)

        source_triangle = []
        target_triangle = []

        for i in range(3):
            source_triangle.append(((source[0][i][0] - source_boundary[0]), (source[0][i][1] - source_boundary[1])))
            target_triangle.append(((target[0][i][0] - target_boundary[0]), (target[0][i][1] - target_boundary[1])))

        source_patch = texture[source_boundary[1]: source_boundary[1] + source_boundary[3], source_boundary[0]: source_boundary[0] + source_boundary[2]]

        transform = cv2.getAffineTransform(np.float32(source_triangle), np.float32(target_triangle))

        target_patch = cv2.warpAffine(source_patch,
                                      transform,
                                      (target_boundary[2], target_boundary[3]),
                                      None,
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101
                                      )

        mask = np.zeros((target_boundary[3], target_boundary[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(target_triangle), (1.0, 1.0, 1.0), 16, 0)

        target_patch = target_patch * mask

        try:
            result[target_boundary[1]: target_boundary[1] + target_boundary[3],
                   target_boundary[0]: target_boundary[0] + target_boundary[2]] = \
                result[target_boundary[1]: target_boundary[1] + target_boundary[3],
                       target_boundary[0]: target_boundary[0] + target_boundary[2]] * ((1.0, 1.0, 1.0) - mask)

            result[target_boundary[1]: target_boundary[1] + target_boundary[3],
                   target_boundary[0]: target_boundary[0] + target_boundary[2]] = \
                result[target_boundary[1]: target_boundary[1] + target_boundary[3],
                       target_boundary[0]: target_boundary[0] + target_boundary[2]] + target_patch
        except:
            pass
    cv2.imwrite(file_name, result)

    return result

def render_animation(texture, points, edges, edge_neighbors, weight, factors, triangles, initial_handles, final_handles, number_of_frames, file_prefix):
    """
    Renders an animation with shape manipulation
    :param texture: initial image as texture
    :param points: list of initial mesh coordinates
    :param edges: set of mesh edges
    :param edge_neighbors: dictionary of all edge neighbors
    :param weight: handle weight
    :param factors: matrix factors from the registration phase
    :param triangles: list of indexed triangles
    :param initial_handles: initial state of handles
    :param final_handles: final state of handles
    :param number_of_frames: number of animated frames
    :param file_prefix: resulting file frame prefix
    """
    dh = np.subtract(final_handles, initial_handles)

    for i in range(number_of_frames + 1):
        current_handles = np.add(initial_handles, np.multiply(float(i) / number_of_frames, dh))
        vertices = compute_new_positions(points, edges, edge_neighbors, current_handles, weight, factors, texture)
        sequence = "{:04}".format(i)
        render = visualize_image(image, points, vertices, triangles, file_name + "_" + sequence + ".png")
        empty = np.zeros_like(render)
        visualize_edges(render, vertices, edges, file_name + "_wired_texture_" + sequence + ".png", current_handles)
        visualize_edges(empty, vertices, edges, file_name + "_wireframe_" + sequence + ".png", current_handles)


###############################
##         Cartoon 7         ##
###############################
# """
file_name = "images/cartoon_7"
image = load_image(file_name + ".png")

boundary = [[40, 664], [86, 576], [93, 490], [129, 331], [92, 386], [59, 396], [25, 364], [40, 310], [88, 314],
            [154, 146], [126, 82], [158, 12], [184, 2], [268, 31], [282, 91], [259, 156], [272, 260], [295, 338],
            [330, 330], [333, 405], [267, 392], [237, 321], [234, 420], [243, 500], [269, 664], [204, 664],
            [185, 543], [173, 430], [156, 564], [120, 664], [40, 664]]

# character = remove_background(image)  # works really bad, forget it for now

triangulation = triangulate(boundary, 20)
visualize_triangulation(image, triangulation, file_name + "_triangulation.png")

points, point_index, edges, edge_neighbors = get_edges_and_neighbors(triangulation)
test_edge_neighbors(edges, edge_neighbors)

print "points:", len(points), "edges:", len(edges)

handles = [[75, 360], [295, 370], [80, 648], [196, 177], [131, 487], [124, 273], [252, 294], [235, 635]]
weight = 1000
factors = registration(points, point_index, edges, edge_neighbors, triangulation)
vertices = compute_new_positions(points, edges, edge_neighbors, handles, weight, factors, image)

triangles = get_indexed_triangles(triangulation, point_index)

initial_handles = [[75, 360], [295, 370], [80, 648], [196, 177], [131, 487], [124, 273], [252, 294], [235, 635]]
final_handles = [[55, 320], [335, 320], [60, 608], [186, 183], [90, 427], [124, 273], [252, 294], [235, 635]]
vertices = compute_new_positions(points, edges, edge_neighbors, final_handles, weight, factors, image)

visualize_image(image, points, vertices, triangles, file_name + "_final.png")
number_of_frames = 30
file_prefix = file_name + "_frame_"
render_animation(image, points, edges, edge_neighbors, weight, factors, triangles, initial_handles, final_handles, number_of_frames, file_prefix)
# """
###############################
##         Cartoon 4         ##
###############################
# """
file_name = "images/cartoon_4_"
image = load_image(file_name + ".png")

boundary = [[224, 507], [95, 507], [79, 461], [115, 410], [47, 352], [60, 291], [146, 179], [224, 163], [274, 213],
            [281, 115], [210, 118], [223, 0], [386, 0], [492, 93], [375, 284], [494, 378], [448, 465], [316, 506], [224, 507]]

# character = remove_background(image)

triangulation = triangulate(boundary, 20)
visualize_triangulation(image, triangulation, file_name + "_triangulation.png")

points, point_index, edges, edge_neighbors = get_edges_and_neighbors(triangulation)
test_edge_neighbors(edges, edge_neighbors)

print "points:", len(points), "edges:", len(edges)

handles = [[191, 221], [213, 406], [78, 338], [328, 298], [230, 94], [416, 143]]
weight = 1000
factors = registration(points, point_index, edges, edge_neighbors, triangulation)
vertices = compute_new_positions(points, edges, edge_neighbors, handles, weight, factors, image)

triangles = get_indexed_triangles(triangulation, point_index)

initial_handles = [[191, 221], [213, 406], [78, 338], [328, 298], [230, 94], [416, 143]]
final_handles = [[149, 228], [258, 382], [30, 366], [336, 270], [126, 159], [436, 223]]
# final_handles = [[191, 221], [213, 406], [78, 338], [328, 298], [230, 94]]
vertices = compute_new_positions(points, edges, edge_neighbors, final_handles, weight, factors, image)

visualize_image(image, points, vertices, triangles, file_name + "_final.png")
number_of_frames = 30
file_prefix = file_name + "_frame_"
render_animation(image, points, edges, edge_neighbors, weight, factors, triangles, initial_handles, final_handles, number_of_frames, file_prefix)
# """

###############################
##         Cartoon 8         ##
###############################
# """
file_name = "images/cartoon_8"
image = load_image(file_name + ".png")

boundary = [[224, 383], [200, 329], [138, 335], [92, 272], [144, 243], [153, 161], [69, 207], [99, 125], [195, 101],
            [285, 0], [364, 19], [363, 154], [441, 231], [435, 304], [372, 209], [367, 301], [275, 353], [320, 383],
            [224, 383]]

# visualize_contour(image, boundary, file_name + "_contour.png")

# character = remove_background(image)

triangulation = triangulate(boundary, 20)
visualize_triangulation(image, triangulation, file_name + "_triangulation.png")

points, point_index, edges, edge_neighbors = get_edges_and_neighbors(triangulation)
test_edge_neighbors(edges, edge_neighbors)

print "points:", len(points), "edges:", len(edges)

handles = [[299, 87], [89, 185], [429, 277], [127, 278], [249, 369]]
weight = 1000
factors = registration(points, point_index, edges, edge_neighbors, triangulation)
vertices = compute_new_positions(points, edges, edge_neighbors, handles, weight, factors, image)

triangles = get_indexed_triangles(triangulation, point_index)

initial_handles = [[299, 87], [89, 185], [429, 277], [127, 278], [249, 369]]
final_handles = [[320, 107], [87, 130], [474, 210], [95, 297], [249, 369]]
# final_handles = [[191, 221], [213, 406], [78, 338], [328, 298], [230, 94]]
vertices = compute_new_positions(points, edges, edge_neighbors, final_handles, weight, factors, image)

visualize_image(image, points, vertices, triangles, file_name + "_final.png")
number_of_frames = 30
file_prefix = file_name + "_frame_"
render_animation(image, points, edges, edge_neighbors, weight, factors, triangles, initial_handles, final_handles, number_of_frames, file_prefix)
# """

