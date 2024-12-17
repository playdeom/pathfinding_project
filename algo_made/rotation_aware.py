import heapq

def rotation_aware_search(n, horizontal_walls, vertical_walls, start, end, turn_cost=0.5):
    def can_move(r1, c1, r2, c2):
        if r2 < 0 or r2 >= n or c2 < 0 or c2 >= n:
            return False
        dr, dc = r2 - r1, c2 - c1
        if dr == 1 and dc == 0:
            return not horizontal_walls[r1+1][c1]
        elif dr == -1 and dc == 0:
            return not horizontal_walls[r1][c1]
        elif dr == 0 and dc == 1:
            return not vertical_walls[r1][c1+1]
        elif dr == 0 and dc == -1:
            return not vertical_walls[r1][c1]
        return False

    def heuristic(r, c):
        return abs(r - end[0]) + abs(c - end[1])

    start_r, start_c = start
    end_r, end_c = end

    directions = [(-1,0),(0,1),(1,0),(0,-1)]
    g_scores = {}
    visited_states = {(start_r, start_c): (None, None)}

    g_scores[(start_r,start_c,None)] = 0
    open_list = []
    heapq.heappush(open_list, (heuristic(start_r,start_c),0,start_r,start_c,None))
    came_from = {}

    while open_list:
        f,g,r,c,d = heapq.heappop(open_list)
        if (r,c) == (end_r,end_c):
            path_nodes = []
            cur = (r,c,d)
            while cur in came_from:
                path_nodes.append((cur[0],cur[1]))
                cur = came_from[cur]
            path_nodes.append((start_r,start_c))
            path_nodes.reverse()
            return path_nodes, visited_states

        if g_scores.get((r,c,d), float('inf')) < g:
            continue

        for ndir,(dr,dc) in enumerate(directions):
            nr,nc = r+dr,c+dc
            if can_move(r,c,nr,nc):
                base_cost = 1
                if d is None:
                    new_cost = g+base_cost
                else:
                    if d == ndir:
                        new_cost = g+base_cost
                    else:
                        new_cost = g+base_cost+turn_cost
                if new_cost < g_scores.get((nr,nc,ndir),float('inf')):
                    g_scores[(nr,nc,ndir)] = new_cost
                    priority = new_cost+heuristic(nr,nc)
                    heapq.heappush(open_list,(priority,new_cost,nr,nc,ndir))
                    came_from[(nr,nc,ndir)] = (r,c,d)
                    if (nr,nc) not in visited_states:
                        visited_states[(nr,nc)] = (r,c)

    return None, visited_states
