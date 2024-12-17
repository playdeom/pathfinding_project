import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from algo_comp.a_star import a_star_search
from algo_comp.flood_fill import flood_fill_search
from algo_comp.dfs_path import dfs_search
from algo_comp.rl_path import rl_path_search
from algo_made.rotation_aware import rotation_aware_search
from algo_made.uf_sps import uf_sps_search
from algo_made.mhip import mhip_pathfinder_search
from algo_made.pco import pso_pathfinding

def generate_perfect_maze(n):
    # Aldous-Broder 알고리즘으로 퍼펙트 미로 생성
    if n % 2 != 0:
        raise ValueError("n은 짝수여야 합니다.")

    visited = [[False]*n for _ in range(n)]
    total_cells = n*n
    visited_count = 1
    horizontal_walls = [[True]*n for _ in range(n+1)]
    vertical_walls = [[True]*(n+1) for _ in range(n)]

    r,c = 0,0
    visited[r][c] = True

    def neighbors(r, c):
        dirs = []
        if r > 0:
            dirs.append((r-1, c))
        if r < n-1:
            dirs.append((r+1, c))
        if c > 0:
            dirs.append((r, c-1))
        if c < n-1:
            dirs.append((r, c+1))
        return dirs

    while visited_count < total_cells:
        nb = neighbors(r,c)
        nr,nc = random.choice(nb)
        if not visited[nr][nc]:
            visited[nr][nc] = True
            visited_count += 1
            dr = nr - r
            dc = nc - c
            if dr == 1 and dc == 0:
                horizontal_walls[r+1][c] = False
            elif dr == -1 and dc == 0:
                horizontal_walls[r][c] = False
            elif dr == 0 and dc == 1:
                vertical_walls[r][c+1] = False
            elif dr == 0 and dc == -1:
                vertical_walls[r][c] = False
        r,c = nr,nc

    return horizontal_walls, vertical_walls

def introduce_loops(n, horizontal_walls, vertical_walls, loop_factor=0.01):
    candidate_walls = []
    for r in range(1, n):
        for c in range(n):
            if horizontal_walls[r][c] == True:
                candidate_walls.append(('h', r, c))
    for r in range(n):
        for c in range(1, n):
            if vertical_walls[r][c] == True:
                candidate_walls.append(('v', r, c))

    remove_count = int(len(candidate_walls)*loop_factor)
    remove_walls = random.sample(candidate_walls, remove_count)

    for wtype, rr, cc in remove_walls:
        if wtype == 'h':
            horizontal_walls[rr][cc] = False
        else:
            vertical_walls[rr][cc] = False

def draw_maze(n, horizontal_walls, vertical_walls, end_path=None, title="Maze"):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')

    # 시작점 (0,0)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='none', zorder=2))
    # 도착점 (n-1, n-1)
    ax.add_patch(Rectangle((n-1, n-1), 1, 1, facecolor='pink', edgecolor='none', zorder=2))

    for r in range(n+1):
        for c in range(n):
            if horizontal_walls[r][c]:
                ax.plot([c,c+1],[r,r], color='black', linewidth=2, zorder=3)
    for r in range(n):
        for c in range(n+1):
            if vertical_walls[r][c]:
                ax.plot([c,c],[r,r+1], color='black', linewidth=2, zorder=3)

    # 최종 경로 (빨강)만 표시
    if end_path is not None:
        px = [c+0.5 for (r,c) in end_path]
        py = [r+0.5 for (r,c) in end_path]
        ax.plot(px, py, color='red', linewidth=2, zorder=4)

    plt.title(title)
    plt.show()

def calculate_moves_and_turns_for_path(path):
    moves = len(path)-1
    turns = 0
    if moves <= 0:
        return 0,0
    directions = []
    for i in range(len(path)-1):
        r1,c1 = path[i]
        r2,c2 = path[i+1]
        dr = r2 - r1
        dc = c2 - c1
        directions.append((dr,dc))
    for i in range(1,len(directions)):
        if directions[i] != directions[i-1]:
            turns += 1
    return moves, turns

def update_average(average_move,average_turn,value_move,value_turn):
    return average_move+value_move,average_turn+value_turn

if __name__ == "__main__":
    a_star_average_moves = 0
    a_star_average_turns = 0
    flood_average_moves = 0
    flood_average_turns = 0
    dfs_average_moves = 0
    dfs_average_turns = 0
    rotation_average_moves_t1 = 0
    rotation_average_turns_t1 = 0
    rotation_average_moves_t05 = 0
    rotation_average_turns_t05 = 0
    # rl_average_moves = 0
    # rl_average_turns = 0
    uf_sps_average_moves = 0
    uf_sps_average_turns = 0
    mhip_average_moves = 0
    mhip_average_turns = 0
    pso_average_moves = 0
    pso_average_turns = 0
    t=5000
    import random
    geteven=list(range(20,101,2))
    import tqdm
    for iter in tqdm.tqdm(range(t)):
        # n = random.choice(geteven)
        n=80
        horizontal_walls, vertical_walls = generate_perfect_maze(n)
        introduce_loops(n, horizontal_walls, vertical_walls, loop_factor=random.randint(0,250)/1000)

        start = (0,0)
        end = (n-1,n-1)

        a_star_path, _ = a_star_search(n, horizontal_walls, vertical_walls, start, end)
        flood_path, _ = flood_fill_search(n, horizontal_walls, vertical_walls, start, end)
        dfs_path_, _ = dfs_search(n, horizontal_walls, vertical_walls, start, end)
        rotation_path_t05, _ = rotation_aware_search(n, horizontal_walls, vertical_walls, start, end, turn_cost=0.5)
        rotation_path_t1, _ = rotation_aware_search(n, horizontal_walls, vertical_walls, start, end, turn_cost=1)
        uf_sps_path, _ = uf_sps_search(n, horizontal_walls, vertical_walls, start, end, turn_cost=0.5)
        mhip_path, _ = mhip_pathfinder_search(n, horizontal_walls, vertical_walls, start, end, turn_cost=0.25)
        pso_path, _ = pso_pathfinding(n, horizontal_walls, vertical_walls, start, end)
        # rl_path, _ = rl_path_search(n, horizontal_walls, vertical_walls, start, end, episodes=2000, alpha=0.1, gamma=0.9, epsilon=0.1)

        a_star_moves, a_star_turns = calculate_moves_and_turns_for_path(a_star_path if a_star_path else [])
        flood_moves, flood_turns = calculate_moves_and_turns_for_path(flood_path if flood_path else [])
        dfs_moves, dfs_turns = calculate_moves_and_turns_for_path(dfs_path_ if dfs_path_ else [])
        rotation_moves_t1, rotation_turns_t1 = calculate_moves_and_turns_for_path(rotation_path_t1 if rotation_path_t1 else [])
        rotation_moves_t05, rotation_turns_t05 = calculate_moves_and_turns_for_path(rotation_path_t05 if rotation_path_t05 else [])
        uf_sps_moves, uf_sps_turns = calculate_moves_and_turns_for_path(uf_sps_path if uf_sps_path else [])
        mhip_moves, mhip_turns = calculate_moves_and_turns_for_path(mhip_path if mhip_path else [])
        pso_moves, pso_turns = calculate_moves_and_turns_for_path(pso_path if pso_path else [])
        # rl_moves, rl_turns = calculate_moves_and_turns_for_path(rl_path if rl_path else [])

        with open("results.txt", "w", encoding="utf-8") as f:
            f.write("[A*]\n")
            f.write("moves={}, turns={}\n\n".format(a_star_moves, a_star_turns))

            f.write("[Flood Fill]\n")
            f.write("moves={}, turns={}\n\n".format(flood_moves, flood_turns))

            f.write("[Rotation aware (turn_cost=0.5)]\n")
            f.write("moves={}, turns={}\n\n".format(rotation_moves_t05, rotation_turns_t05))

            f.write("[Rotation aware (turn_cost=1)]\n")
            f.write("moves={}, turns={}\n\n".format(rotation_moves_t1, rotation_turns_t1))

            f.write("[UF SPS]\n")
            f.write("moves={}, turns={}\n\n".format(uf_sps_moves, uf_sps_turns))

            f.write("[MHIP]\n")
            f.write("moves={}, turns={}\n\n".format(mhip_moves, mhip_turns))

            f.write("[PSO]\n")
            f.write("moves={}, turns={}\n\n".format(pso_moves, pso_turns))

            f.write("[DFS]\n")
            f.write("moves={}, turns={}\n\n".format(dfs_moves, dfs_turns))

            # f.write("[RL Learning]\n")
            # f.write("최종경로: 이동={}, 턴={}\n\n".format(rl_moves, rl_turns))

        a_star_average_moves,a_star_average_turns=update_average(a_star_average_moves,a_star_average_turns,a_star_moves,a_star_turns)
        flood_average_moves,flood_average_turns=update_average(flood_average_moves,flood_average_turns,flood_moves,flood_turns)
        rotation_average_moves_t05,rotation_average_turns_t05=update_average(rotation_average_moves_t05,rotation_average_turns_t05,rotation_moves_t05,rotation_turns_t05)
        rotation_average_moves_t1,rotation_average_turns_t1=update_average(rotation_average_moves_t1,rotation_average_turns_t1,rotation_moves_t1,rotation_turns_t1)
        uf_sps_average_moves,uf_sps_average_turns=update_average(uf_sps_average_moves,uf_sps_average_turns,uf_sps_moves,uf_sps_turns)
        mhip_average_moves,mhip_average_turns=update_average(mhip_average_moves,mhip_average_turns,mhip_moves,mhip_turns)
        pso_average_moves,pso_average_turns=update_average(pso_average_moves,pso_average_turns,pso_moves,pso_turns)
        dfs_average_moves,dfs_average_turns=update_average(dfs_average_moves,dfs_average_turns,dfs_moves,dfs_turns)
        
        # if rl_moves != 0:
        #     rl_average_moves += rl_moves
        #     rl_average_turns += rl_turns
    with open(f'/data/average_res_{n}_{t}.txt','w',encoding='utf-8') as f:
        f.write("iterations: {}\n".format(t))
        f.write("maze size: {}\n\n".format(n))
        
        f.write("[A*]\n")
        f.write("average moves={}, turns={}\n\n".format(a_star_average_moves/t, a_star_average_turns/t))
        
        f.write("[Flood Fill]\n")
        f.write("average moves={}, turns={}\n\n".format(flood_average_moves/t, flood_average_turns/t))
        
        f.write("[Rotation aware (turn_cost=0.5)]\n")
        f.write("average moves={}, turns={}\n\n".format(rotation_average_moves_t05/t, rotation_average_turns_t05/t))
        
        f.write("[Rotation aware (turn_cost=1)]\n")
        f.write("average moves={}, turns={}\n\n".format(rotation_average_moves_t1/t, rotation_average_turns_t1/t))
        
        f.write("[UF SPS]\n")
        f.write("average moves={}, turns={}\n\n".format(uf_sps_average_moves/t, uf_sps_average_turns/t))
        
        f.write("[MHIP]\n")
        f.write("average moves={}, turns={}\n\n".format(mhip_average_moves/t, mhip_average_turns/t))
        
        f.write("[PSO]\n")
        f.write("average moves={}, turns={}\n\n".format(pso_average_moves/t, pso_average_turns/t))
        
        f.write("[DFS]\n")
        f.write("average moves={}, turns={}\n\n".format(dfs_average_moves/t, dfs_average_turns/t))
        
        # f.write("[RL Learning]\n")
        # f.write("최종경로: 이동={}, 턴={}\n\n".format(rl_average_moves/t, rl_average_turns/t))
        
    # 시각화 (최종 경로만 표시)
    draw_maze(n, horizontal_walls, vertical_walls, end_path=pso_path, title="PSO Path with Loops")
    # draw_maze(n, horizontal_walls, vertical_walls, end_path=rotation_path_t1, title="Rotation-Aware Path with Loops")
    # draw_maze(n, horizontal_walls, vertical_walls, end_path=a_star_path, title="A* Path with Loops")
    # draw_maze(n, horizontal_walls, vertical_walls, end_path=flood_path, title="Flood Fill Path with Loops")
    # draw_maze(n, horizontal_walls, vertical_walls, end_path=dfs_path_, title="DFS Path with Loops")
    # draw_maze(n, horizontal_walls, vertical_walls, end_path=rl_path, title="RL Path with Loops")
