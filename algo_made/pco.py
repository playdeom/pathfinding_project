import random
from collections import deque

def pso_pathfinding(n, horizontal_walls, vertical_walls, start, end, swarm_size=50, iterations=200, inertia_weight=0.9, inertia_min=0.4, inertia_max=0.9, cognitive_coeff=2.0, social_coeff=2.0, velocity_clamp=2):
    """
    개선된 입자 군집 최적화 기반 경로 탐색 알고리즘 (Enhanced PSOPF)
    
    Parameters:
    - n: 미로의 크기 (n x n)
    - horizontal_walls: 수평 벽 배열 (n+1 x n)
    - vertical_walls: 수직 벽 배열 (n x n+1)
    - start: 시작점 (r, c)
    - end: 목표점 (r, c)
    - swarm_size: 입자 군집의 크기
    - iterations: 최대 반복 횟수
    - inertia_weight: 초기 관성 계수
    - inertia_min: 최소 관성 계수
    - inertia_max: 최대 관성 계수
    - cognitive_coeff: 개인 최적 계수
    - social_coeff: 군집 최적 계수
    - velocity_clamp: 속도 클램핑 범위
    
    Returns:
    - best_path: 최종 경로 리스트 [(r1, c1), (r2, c2), ...]
    - visited_states: 방문한 상태의 부모 정보 {(r, c): (pr, pc), ...}
    """
    
    # 방향 정의: 상, 우, 하, 좌
    DIRECTIONS = [(-1,0),(0,1),(1,0),(0,-1)]
    
    class Particle:
        def __init__(self, position):
            self.position = position  # (r, c)
            self.velocity = [0, 0]    # (dr, dc)
            self.best_position = position
            self.best_cost = float('inf')
            self.path = [position]
    
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
    
    def calculate_cost(path):
        """
        경로의 총 비용 계산 (이동 거리 + 회전 횟수)
        """
        if not path:
            return float('inf')
        moves = len(path) - 1
        turns = 0
        if moves > 0:
            prev_dir = None
            for i in range(1, len(path)):
                dr = path[i][0] - path[i-1][0]
                dc = path[i][1] - path[i-1][1]
                try:
                    current_dir = DIRECTIONS.index((dr, dc))
                except ValueError:
                    current_dir = None
                if prev_dir is not None and current_dir != prev_dir:
                    turns += 1
                prev_dir = current_dir
        return moves + turns * 0.5  # 회전 비용 가중치
    
    def get_neighbors(r, c):
        neighbors = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if can_move(r, c, nr, nc):
                neighbors.append((nr, nc))
        return neighbors
    
    def reconstruct_path(parents, end):
        """
        BFS를 사용하여 경로 복원
        """
        path = []
        current = end
        while current in parents:
            path.append(current)
            current = parents[current]
        path.append(start)
        path.reverse()
        return path
    
    # Initialize swarm
    swarm = [Particle(start) for _ in range(swarm_size)]
    global_best_position = start
    global_best_cost = float('inf')
    
    for iteration in range(iterations):
        # Adapt inertia weight
        inertia = inertia_min + (inertia_max - inertia_min) * (1 - iteration / iterations)
        
        for particle in swarm:
            # Update velocity
            new_velocity = [
                inertia * particle.velocity[0] +
                cognitive_coeff * random.random() * (particle.best_position[0] - particle.position[0]) +
                social_coeff * random.random() * (global_best_position[0] - particle.position[0]),
                inertia * particle.velocity[1] +
                cognitive_coeff * random.random() * (particle.best_position[1] - particle.position[1]) +
                social_coeff * random.random() * (global_best_position[1] - particle.position[1])
            ]
            # Velocity clamping
            new_velocity = [max(-velocity_clamp, min(velocity_clamp, v)) for v in new_velocity]
            particle.velocity = new_velocity
            
            # Update position
            new_r = particle.position[0] + int(round(particle.velocity[0]))
            new_c = particle.position[1] + int(round(particle.velocity[1]))
            
            # Ensure new position is within bounds and not hitting walls
            if can_move(particle.position[0], particle.position[1], new_r, new_c):
                particle.position = (new_r, new_c)
                particle.path.append(particle.position)
                cost = calculate_cost(particle.path)
                if cost < particle.best_cost:
                    particle.best_cost = cost
                    particle.best_position = particle.position
                    # Update global best
                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best_position = particle.position
        
        # Check for convergence: if global best cost hasn't improved for a certain number of iterations
        # (Optional: Implement if needed)
        
        # Early stopping if global best reaches the end
        if global_best_position == end:
            break
    
    # Reconstruct path from the best particle
    best_particle = min(swarm, key=lambda p: calculate_cost(p.path))
    if best_particle.position == end:
        path = best_particle.path
    else:
        # If no particle reached the end, perform BFS from the closest particle
        closest_particles = sorted(swarm, key=lambda p: calculate_cost(p.path))[:5]
        closest_position = closest_particles[0].position
        queue = deque([closest_position])
        parents = {closest_position: None}
        while queue:
            current = queue.popleft()
            if current == end:
                break
            for neighbor in get_neighbors(current[0], current[1]):
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)
        path = reconstruct_path(parents, end) if end in parents else None
    
    # Path smoothing
    def smooth_path(path):
        if not path or len(path) < 3:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            advanced = False
            while j > i + 1:
                # smoothed[-1]에서 path[j]로 바로 갈 수 있는지 확인
                start_cell = smoothed[-1]
                end_cell = path[j]

                # 같은 행이나 같은 열에 있을 때만 직선 경로 확인 가능
                if start_cell[0] == end_cell[0] or start_cell[1] == end_cell[1]:
                    # 행 또는 열이 동일한 경우
                    clear = True
                    if start_cell[0] == end_cell[0]:
                        # 같은 행, 열을 증감하며 확인
                        r = start_cell[0]
                        c1, c2 = start_cell[1], end_cell[1]
                        step = 1 if c2 > c1 else -1
                        c = c1
                        while c != c2:
                            next_c = c + step
                            if not can_move(r, c, r, next_c):
                                clear = False
                                break
                            c = next_c
                    else:
                        # 같은 열, 행을 증감하며 확인
                        c = start_cell[1]
                        r1, r2 = start_cell[0], end_cell[0]
                        step = 1 if r2 > r1 else -1
                        r = r1
                        while r != r2:
                            next_r = r + step
                            if not can_move(r, c, next_r, c):
                                clear = False
                                break
                            r = next_r

                    if clear:
                        smoothed.append(end_cell)
                        i = j
                        advanced = True
                        break
                j -= 1

            if not advanced:
                # 더 멀리 점프 불가, 바로 다음 칸 추가
                if i + 1 < len(path):
                    smoothed.append(path[i+1])
                i += 1

        return smoothed

    
    # if path:
        # path = smooth_path(path)
    
    # Generate visited_states for visualization
    visited_states = {}
    if path:
        for i in range(1, len(path)):
            visited_states[path[i]] = path[i-1]
    
    return path, visited_states
