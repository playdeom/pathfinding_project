# mhip.py

import heapq

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx

# 방향 정의: 0=북(상), 1=동(우), 2=남(하), 3=서(좌)
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def multi_heuristic(x, y, current_direction, goal_x, goal_y, turn_cost=0.5):
    # 맨해튼 거리 계산
    manhattan_distance = abs(x - goal_x) + abs(y - goal_y)
    
    # 유클리드 거리 계산
    euclidean_distance = ((x - goal_x)**2 + (y - goal_y)**2)**0.5
    
    # 목표 방향 계산
    if goal_x > x:
        goal_direction = 1  # 동쪽
    elif goal_x < x:
        goal_direction = 3  # 서쪽
    elif goal_y > y:
        goal_direction = 2  # 남쪽
    else:
        goal_direction = 0  # 북쪽
    
    # 회전 비용 계산 (최소 회전 횟수)
    turn_diff = min((current_direction - goal_direction) % 4, (goal_direction - current_direction) % 4)
    
    # 종합 휴리스틱 비용 (가중치 조정 가능)
    heuristic_cost = manhattan_distance + 0.5 * turn_diff + 0.2 * euclidean_distance
    return heuristic_cost

def mhip_pathfinder_search(n, horizontal_walls, vertical_walls, start, end, turn_cost=0.5):
    """
    다중 휴리스틱 통합 경로 탐색 알고리즘 (MHIP)
    
    Parameters:
        n (int): 미로의 크기 (n x n)
        horizontal_walls (list of list of bool): 수평 벽 정보
        vertical_walls (list of list of bool): 수직 벽 정보
        start (tuple): 시작 위치 (r, c)
        end (tuple): 목표 위치 (r, c)
        turn_cost (float): 회전 비용 가중치
    
    Returns:
        tuple:
            path (list of tuples): 최종 경로 (리스트 형태)
            visited_states (dict): 경로 복원을 위한 부모 노드 정보 { (r,c): (pr, pc) }
    """
    start_r, start_c = start
    end_r, end_c = end

    uf = UnionFind(n * n)

    def cell_id(r, c):
        return r * n + c

    def can_move(r1, c1, r2, c2):
        if r2 < 0 or r2 >= n or c2 < 0 or c2 >= n:
            return False
        dr = r2 - r1
        dc = c2 - c1
        if dr == 1 and dc == 0:
            return not horizontal_walls[r1+1][c1]
        elif dr == -1 and dc == 0:
            return not horizontal_walls[r1][c1]
        elif dr == 0 and dc == 1:
            return not vertical_walls[r1][c1+1]
        elif dr == 0 and dc == -1:
            return not vertical_walls[r1][c1]
        return False

    open_list = []
    heapq.heapify(open_list)

    visited_states = {(start_r, start_c): (None, None)}  # (r,c) : (pr, pc)

    # 시작점에서 모든 방향으로 시도
    for d in range(4):
        h = multi_heuristic(start_r, start_c, d, end_r, end_c, turn_cost)
        f = h
        g = 0
        heapq.heappush(open_list, (f, g, start_r, start_c, d))

    visited = set()
    parent = {}  # (r,c,d) : (pr, pc, pd)

    while open_list:
        f, g, r, c, d = heapq.heappop(open_list)

        if (r, c, d) in visited:
            continue
        visited.add((r, c, d))

        if (r, c) == (end_r, end_c):
            # 경로 복원
            path = []
            current = (r, c, d)
            while current in parent:
                path.append((current[0], current[1]))
                current = parent[current]
            path.append((start_r, start_c))
            path.reverse()
            return path, visited_states

        # 가능한 행동: 전진(forward), 좌회전+전진(left), 우회전+전진(right)
        for action in ['forward', 'left', 'right']:
            if action == 'forward':
                nd = d
                cost = 1  # 이동 비용
            elif action == 'left':
                nd = (d - 1) % 4
                cost = 1 + turn_cost  # 이동 + 좌회전 비용
            elif action == 'right':
                nd = (d + 1) % 4
                cost = 1 + turn_cost  # 이동 + 우회전 비용

            dr, dc = directions[nd]
            nr, nc = r + dr, c + dc

            # 미로 범위 내 확인 및 벽 체크
            if can_move(r, c, nr, nc):
                # 전진 시 유니온 파인드로 연결
                if action == 'forward':
                    uf.union(cell_id(r, c), cell_id(nr, nc))

                # 연결성 확인
                connected = uf.find(cell_id(r, c)) == uf.find(cell_id(nr, nc))

                # 회전 최소화를 위한 추가 비용
                if connected and action == 'forward':
                    priority_cost = 0  # 직선 경로 유지
                else:
                    priority_cost = 1  # 회전 필요 시 추가 비용

                # 새로운 g, f 계산
                new_g = g + cost + priority_cost
                new_h = multi_heuristic(nr, nc, nd, end_r, end_c, turn_cost)
                new_f = new_g + new_h

                # 상태가 방문되지 않았다면 추가
                if (nr, nc, nd) not in visited:
                    heapq.heappush(open_list, (new_f, new_g, nr, nc, nd))
                    parent[(nr, nc, nd)] = (r, c, d)
                    if (nr, nc) not in visited_states:
                        visited_states[(nr, nc)] = (r, c)

    # 경로 없음
    return None, visited_states
