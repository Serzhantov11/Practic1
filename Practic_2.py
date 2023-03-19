import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import math
import random
from collections import deque, defaultdict
import heapq

class Graph(object):
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance

graph = Graph()

def generate_grap(img, graph):
    positions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    dost_pos = [0]

    # Проверка возможности перехода в точку с учётом смезения
    def is_posible(pos, bias, border):
        if 0 <= pos[0] + bias[0] < border[0] and 0 <= pos[1] + bias[1] < border[1]:
            return True
        return False

    # Получение веса перехода в точку
    def weights(bias, value, pos):
        """Вес стоимости перемещения"""
        if 0 in bias and value in dost_pos:
            return 1
        elif 0 not in bias and value in dost_pos:
            if bias == positions[4] and img[pos[0] + positions[0][0]][pos[1] + positions[0][1]] == 1 and \
                    img[pos[0] + positions[2][0]][pos[1] + positions[2][1]] == 1:
                return 100
            elif bias == positions[5] and img[pos[0] + positions[2][0]][pos[1] + positions[2][1]] == 1 and \
                    img[pos[0] + positions[1][0]][pos[1] + positions[1][1]] == 1:
                return 100
            elif bias == positions[6] and img[pos[0] + positions[3][0]][pos[1] + positions[3][1]] == 1 and \
                    img[pos[0] + positions[0][0]][pos[1] + positions[0][1]] == 1:
                return 100
            elif bias == positions[7] and img[pos[0] + positions[1][0]][pos[1] + positions[1][1]] == 1 and \
                    img[pos[0] + positions[3][0]][pos[1] + positions[3][1]] == 1:
                return 100
            else:
                return math.sqrt(2)
        elif bias in positions and value not in dost_pos:
            return 100

    # Инициализация графа и заполнение его вершинами
    graph = graph
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            graph.add_node((i, j))

    # формирование связей графа
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for pos in positions:
                if is_posible((i, j), pos, (img.shape[0], img.shape[1])):
                    graph.add_edge((i, j), (i + pos[0], j + pos[1]), weights(pos, img[i][j], (i, j)))

    return graph

def generate_image():
    # Генерация карты случайного размера в диапазоне 35 * 45
    # Заполнение карты случайными препятсвиями с случайной частотой
    # Генерация начальной и финишной точки до которой должен проследовать робот


    img = np.zeros((np.random.randint(35, 45), np.random.randint(35, 45)))

    for i in range(random.randint(70, 85)):
        img[random.randint(0, img.shape[0] - 1)][random.randint(0, img.shape[1] - 1)] = 1
    k = 2
    while k != 1:
        x, y = random.randint(img.shape[0] // 2, img.shape[0] - 1), random.randint(img.shape[1] // 2, img.shape[1] - 1)
        if img[x][y] != 0:
            start = (x, y)
            k -= 1
    end = (random.randint(1, 10), random.randint(1, 10))
    return start, end, img

def dijkstra(graph, start):
    visited = {start: 0}
    path = {}

    nodes = set(graph.nodes)

    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node
        if min_node is None:
            break

        nodes.remove(min_node)
        current_weight = visited[min_node]

        for edge in graph.edges[min_node]:
            try:
                weight = current_weight + graph.distances[(min_node, edge)]
            except:
                continue
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node

    return visited, path


def shortest_path(graph, origin, destination):
    visited, paths = dijkstra(graph, origin)
    full_path = deque()
    _destination = paths[destination]

    while _destination != origin:
        full_path.appendleft(_destination)
        _destination = paths[_destination]

    full_path.appendleft(origin)
    full_path.append(destination)

    return visited[destination], list(full_path)

def astar(start, goal, obstacles):

    def heuristic(a, b):
        return abs(b[0] - a[0]) + abs(b[1] - a[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while open_set:

        current = heapq.heappop(open_set)[1]

        if current == goal:
            break

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_node = (current[0] + dx, current[1] + dy)

            if next_node[0] < 0 or next_node[0] >= len(obstacles) or next_node[1] < 0 or next_node[1] >= len(obstacles[0]) or obstacles[next_node[0]][next_node[1]]:
                continue

            new_cost = cost_so_far[current] + 1

            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(goal, next_node)
                heapq.heappush(open_set, (priority, next_node))
                came_from[next_node] = current

    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()

    return path

def main():
    start, end, IMG = generate_image()
    graph = Graph()
    graph = generate_grap(IMG, graph)
    map_orig = IMG.copy()
    map_orig[start[0]][start[1]] = 3
    map_orig[end[0]][end[1]] = 3
    mapp_d = IMG.copy()
    mapp_a = IMG.copy()
    mapp_a = IMG.copy() == 1
    pst = astar(start, end, mapp_a)
    pathh = shortest_path(graph, start, end)
    for point in pathh[1]:
        mapp_d[point[0]][point[1]] = 5

    mapp_a = IMG.copy()
    for point in list(pst):
        mapp_a[point[0]][point[1]] = 5

    ploters = [map_orig, mapp_d, mapp_a]
    tittles = ['Оригинальное изображение', 'Дейкстра', 'Астар']
    print(f'start x {start[1]} -> end x {end[1]}, start y {start[0]} -> end y {end[0]}')
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(ploters[i], cmap='gray')
        plt.title(tittles[i])

    plt.show()

main()
