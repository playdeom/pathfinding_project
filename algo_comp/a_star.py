import heapq

def a_star_search(n, horizontal_walls, vertical_walls, start, end):
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

    open_list = []
    heapq.heappush(open_list, (heuristic(start_r, start_c), 0, start_r, start_c))
    came_from = {}
    g_score = {(start_r, start_c): 0}
    visited_states = {(start_r, start_c): (None, None)}

    while open_list:
        _, cost, r, c = heapq.heappop(open_list)
        if (r, c) == (end_r, end_c):
            path = []
            cur = (end_r, end_c)
            while cur != (start_r, start_c):
                path.append(cur)
                cur = came_from[cur]
            path.append((start_r, start_c))
            path.reverse()
            return path, visited_states

        if g_score.get((r, c), float('inf')) < cost:
            continue

        for nr, nc in [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]:
            if can_move(r, c, nr, nc):
                new_cost = cost + 1
                if (nr, nc) not in g_score or new_cost < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = new_cost
                    priority = new_cost + heuristic(nr, nc)
                    heapq.heappush(open_list, (priority, new_cost, nr, nc))
                    came_from[(nr, nc)] = (r, c)
                    if (nr,nc) not in visited_states:
                        visited_states[(nr, nc)] = (r, c)

    return None, visited_states
